from dataclasses import dataclass
from pathlib import Path
import ctypes
import hashlib
import os
import re
import shutil
import tempfile
import threading

from triton.backends.compiler import (
    BaseBackend,
    GPUTarget,
    Language,
)
from triton._C.libtriton import ir, passes


BIREN_TRIPLE = "biren-biren-supa"
BIREN_DATALAYOUT = (
    "e-p:64:64-"
    "p1:64:64:64:64-"
    "p2:32:32-"
    "p3:64:64:64:64-"
    "p4:32:32-"
    "p32:32:32-"
    "p33:32:32-"
    "p34:32:32-"
    "p35:32:32-"
    "p36:32:32-"
    "p63:32:32-"
    "p64:32:32-"
    "p65:32:32-"
    "p66:32:32-"
    "p67:32:32-"
    "p68:32:32-"
    "i64:64-"
    "v16:16-v24:32-v32:32-"
    "v64:64-v96:128-v128:128-"
    "v256:256-"
    "n16:32:64-S32-A4-ni:7"
)


def finalize_supa_llvm(text, kernel_name, metadata, options):
    """Rewrite FlagTree LLVM text into Biren-facing LLVM.

    FlagTree LLVM does not need to understand Biren intrinsics or
    `biren_supa_kernel`. Placeholders are rewritten here. Only the
    Triton entry function is marked as a kernel.
    """
    del metadata, options

    if re.search(r"^target triple = ", text, flags=re.MULTILINE):
        text = re.sub(
            r"^target triple = .*?$",
            f'target triple = "{BIREN_TRIPLE}"',
            text,
            flags=re.MULTILINE,
        )
    else:
        text = f'target triple = "{BIREN_TRIPLE}"\n' + text

    if re.search(r"^target datalayout = ", text, flags=re.MULTILINE):
        text = re.sub(
            r"^target datalayout = .*?$",
            f'target datalayout = "{BIREN_DATALAYOUT}"',
            text,
            flags=re.MULTILINE,
        )
    else:
        text = f'target datalayout = "{BIREN_DATALAYOUT}"\n' + text

    text = text.replace(
        "@__supa_get_group_id_x",
        "@llvm.biren.get.group.id.x",
    )
    text = text.replace(
        "@__supa_get_group_id_y",
        "@llvm.biren.get.group.id.y",
    )
    text = text.replace(
        "@__supa_get_group_id_z",
        "@llvm.biren.get.group.id.z",
    )
    text = text.replace(
        "@__supa_get_local_id_x",
        "@llvm.biren.get.local.id.x",
    )

    pattern = (
        rf"(define\s+(?:dso_local\s+)?)"
        rf"void\s+@{re.escape(kernel_name)}\("
    )
    replacement = (
        rf"\1biren_supa_kernel void @{kernel_name}("
    )
    text, count = re.subn(pattern, replacement, text)
    if count != 1:
        raise RuntimeError(
            "Expected exactly one kernel entry "
            f"{kernel_name!r}, decorated {count}"
        )
    return text


def _scalarize_singleton_structs(text):
    """Split ``{ T, T, ... }`` aggregates into scalar values.

    A one-field rewrite of ``{ float, float }`` printed ``phi float, float``,
    which BirenLLC rejects. Each field becomes its own ``phi``.
    """
    fields_of = {}
    fresh_n = 0

    def fresh():
        nonlocal fresh_n
        fresh_n += 1
        return f"%agg{fresh_n}"

    def zero_for(ty):
        return "0.000000e+00" if ty == "float" else "0"

    def struct_fields(ty):
        return [part.strip() for part in ty.split(",")]

    insert_re = re.compile(
        r"^(\s*)(%[\w.]+) = insertvalue \{ ([^{}]+) \} (%[\w.]+|undef), (\S+) (%[\w.]+), (\d+)\s*$"
    )
    phi_re = re.compile(
        r"^(\s*)(%[\w.]+) = phi \{ ([^{}]+) \} (.+)\s*$"
    )
    extract_re = re.compile(
        r"^(\s*)(%[\w.]+) = extractvalue \{ ([^{}]+) \} (%[\w.]+), (\d+)\s*$"
    )
    # Insertvalue chains are collected first. A loop header phi is
    # textually above the insertvalue that feeds its back edge.
    for line in text.splitlines():
        insert = insert_re.match(line)
        if not insert:
            continue
        dest, agg_ty, base, index = (
            insert.group(2),
            insert.group(3),
            insert.group(4),
            int(insert.group(7)),
        )
        fields = struct_fields(agg_ty)
        if base == "undef":
            slots = [zero_for(ty) for ty in fields]
        else:
            slots = list(fields_of.get(base, []))
        if len(slots) != len(fields) or not (0 <= index < len(fields)):
            continue
        slots[index] = insert.group(6)
        fields_of[dest] = slots

    out = []
    for line in text.splitlines():
        if insert_re.match(line) and insert_re.match(line).group(2) in fields_of:
            continue

        phi = phi_re.match(line)
        if phi:
            indent, dest, agg_ty, incoming = (
                phi.group(1),
                phi.group(2),
                phi.group(3),
                phi.group(4),
            )
            fields = struct_fields(agg_ty)
            parts = re.findall(
                r"\[\s*([^,\]]+)\s*,\s*(%[\w.]+)\s*\]",
                incoming,
            )
            columns = []
            ok = True
            for value, _block in parts:
                value = value.strip()
                if value == "zeroinitializer":
                    columns.append([zero_for(ty) for ty in fields])
                elif value in fields_of and len(fields_of[value]) == len(fields):
                    columns.append(fields_of[value])
                else:
                    ok = False
                    break
            if not ok or len(columns) != len(parts):
                out.append(line)
                continue
            # Keep the original numbered name on the first field. Dropping
            # it makes the next instruction skip a slot ("%75" vs "%74").
            names = [dest] + [fresh() for _ in fields[1:]]
            fields_of[dest] = names
            for index, name in enumerate(names):
                incoming_text = ", ".join(
                    f"[ {columns[src][index]}, {block} ]"
                    for src, (_value, block) in enumerate(parts)
                )
                out.append(f"{indent}{name} = phi {fields[index]} {incoming_text}")
            continue

        extract = extract_re.match(line)
        if extract:
            dest, src, index = extract.group(2), extract.group(4), int(extract.group(5))
            slots = fields_of.get(src)
            if slots is None or not (0 <= index < len(slots)):
                out.append(line)
                continue
            fields_of[dest] = [slots[index]]
            # A one-element name is inlined at its uses below.
            if slots[index].startswith("%"):
                fields_of[dest] = [slots[index]]
            else:
                out.append(f"{extract.group(1)}{dest} = add {extract.group(3).split(',')[0].strip()} {slots[index]}, 0")
                fields_of[dest] = [dest]
            continue

        out.append(line)

    text = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    for name, slots in fields_of.items():
        if len(slots) == 1 and slots[0] != name and slots[0].startswith("%"):
            text = re.sub(
                rf"(?<![\w.]){re.escape(name)}(?![\w.])",
                slots[0],
                text,
            )
    return text


def _lower_bfloat_casts(text):
    """Replace bfloat fpext/fptrunc. The loader does not convert them.

    BirenLLC numbers ``%0``, ``%1``, ``%2`` in order. A later ``%226``
    inserted in the middle is rejected ("expected to be numbered '%85'").
    Temporaries use non-numeric names so they do not take a slot.
    """
    nxt = 0

    def fresh():
        nonlocal nxt
        nxt += 1
        return f"%bf{nxt}"

    def fpext(match):
        indent, dst, src = match.group(1), match.group(2), match.group(3)
        i16, i32, shifted = fresh(), fresh(), fresh()
        return (
            f"{indent}{i16} = bitcast bfloat {src} to i16\n"
            f"{indent}{i32} = zext i16 {i16} to i32\n"
            f"{indent}{shifted} = shl i32 {i32}, 16\n"
            f"{indent}{dst} = bitcast i32 {shifted} to float"
        )

    def fptrunc(match):
        indent, dst, src = match.group(1), match.group(2), match.group(3)
        # Round-to-nearest-even. Truncating the low 16 bits is biased and
        # accumulates across every BF16 store. This torch build turns every
        # NaN into 0xFFFF, which is still a NaN rather than an Inf.
        bits, abs_bits, is_nan, hi0, lsb, bias, rounded, hi, raw, out16 = (
            fresh() for _ in range(10)
        )
        return "\n".join([
            f"{indent}{bits} = bitcast float {src} to i32",
            f"{indent}{abs_bits} = and i32 {bits}, 2147483647",
            f"{indent}{is_nan} = icmp ugt i32 {abs_bits}, 2139095040",
            f"{indent}{hi0} = lshr i32 {bits}, 16",
            f"{indent}{lsb} = and i32 {hi0}, 1",
            f"{indent}{bias} = add i32 {lsb}, 32767",
            f"{indent}{rounded} = add i32 {bits}, {bias}",
            f"{indent}{hi} = lshr i32 {rounded}, 16",
            f"{indent}{raw} = trunc i32 {hi} to i16",
            f"{indent}{out16} = select i1 {is_nan}, i16 -1, i16 {raw}",
            f"{indent}{dst} = bitcast i16 {out16} to bfloat",
        ])

    text = re.sub(
        r"^(\s*)(%[\w.]+) = fpext bfloat (%[\w.]+) to float\s*$",
        fpext,
        text,
        flags=re.M,
    )
    text = re.sub(
        r"^(\s*)(%[\w.]+) = fptrunc float (%[\w.]+) to bfloat\s*$",
        fptrunc,
        text,
        flags=re.M,
    )
    return text


def _renumber_numeric_values(text):
    """Renumber ``%0, %1, %2`` after instructions are inserted or deleted.

    BirenLLC requires each numbered value, including block labels, to be
    the next integer in source order. Named values (``%agg1``, ``%bf1``)
    do not take a slot.
    """

    def renumber_function(func):
        header, sep, body = func.partition("{")
        if not sep:
            return func
        order = []
        seen = set()

        def add(number):
            if number not in seen:
                seen.add(number)
                order.append(number)

        for match in re.finditer(r"%(\d+)\b", header):
            add(match.group(1))
        # An entry block with no label still consumes the next number.
        # Six arguments are %0-%5, so the first instruction is %7, not %6.
        body_lines = body.splitlines()
        first = next((line for line in body_lines if line.strip()), "")
        if not re.match(r"\s*\d+:", first):
            add(str(len(order)))
        for line in body_lines:
            label = re.match(r"\s*(\d+):", line)
            if label:
                add(label.group(1))
                continue
            inst = re.match(r"\s*%(\d+)\s*=", line)
            if inst:
                add(inst.group(1))
        mapping = {old: str(i) for i, old in enumerate(order)}
        if all(mapping[key] == key for key in mapping):
            return func

        def sub_value(match):
            return "%" + mapping.get(match.group(1), match.group(1))

        new_header = re.sub(r"%(\d+)\b", sub_value, header)
        new_lines = []
        for line in body.splitlines(keepends=True):
            line = re.sub(
                r"^(\s*)(\d+):",
                lambda match: match.group(1)
                + mapping.get(match.group(2), match.group(2))
                + ":",
                line,
                count=1,
            )
            line = re.sub(r"%(\d+)\b", sub_value, line)
            new_lines.append(line)
        return new_header + sep + "".join(new_lines)

    parts = re.split(r"(?=^define )", text, flags=re.M)
    return "".join(
        renumber_function(part) if part.startswith("define ") else part
        for part in parts
    )


def _narrow_i64_index_math(text):
    """Rewrite i64 arithmetic to i32, but keep 8-byte int64 loads.

    ``mul``/``select`` on i64 makes ``sudrvModuleGetFunction`` hang.
    SUPA ``torch.long`` values are 8 bytes apart. Keep ``getelementptr i64``
    so the address scales by 8, then load the low 4 bytes. Arithmetic stays
    i32 because i64 ``mul``/``select`` does not come back from the loader.
    """
    lines = text.splitlines()
    insts = []
    for line in lines:
        match = re.match(r"^(\s*)(%[\w.]+) = (.*\S)\s*$", line)
        if match:
            insts.append((match.group(2), match.group(3)))

    def fits_i32(token):
        if re.fullmatch(r"-?\d+", token):
            value = int(token)
            return -(2**31) <= value < 2**31
        return False

    by_name = {name: rhs for name, rhs in insts}
    narrow = set()
    changed = True
    while changed:
        changed = False

        def is_narrow(token):
            return token in narrow or fits_i32(token)

        for name, rhs in insts:
            if name in narrow:
                continue
            ok = False
            gep = re.match(
                r"getelementptr i64, ptr addrspace\(\d+\) %[\w.]+, (i32|i64) (%[\w.]+|-?\d+)$",
                rhs,
            )
            if gep and gep.group(1) == "i32":
                # The pointer is still i64-strided; the loaded value is the
                # candidate. Mark the load, not the pointer.
                pass
            elif gep and gep.group(1) == "i64" and is_narrow(gep.group(2)):
                pass
            load = re.match(
                r"load i64, (ptr addrspace\(\d+\) (%[\w.]+)), align \d+$",
                rhs,
            )
            if load:
                ptr = load.group(2)
                ptr_rhs = by_name.get(ptr, "")
                if ptr_rhs.startswith("getelementptr i64,"):
                    ok = True
            elif rhs.startswith("phi i64 "):
                incoming = re.findall(
                    r"\[\s*([^,\]]+)\s*,\s*%[\w.]+\s*\]", rhs
                )
                ok = bool(incoming) and all(
                    is_narrow(part.strip()) for part in incoming
                )
            elif re.match(r"(?:add|sub|mul|shl|lshr|ashr|and|or|xor) i64 ", rhs):
                ops = re.findall(r"(%[\w.]+|-?\d+)", rhs)
                ok = len(ops) >= 2 and all(is_narrow(op) for op in ops)
            elif rhs.startswith("icmp ") and " i64 " in rhs:
                ops = re.findall(r"(%[\w.]+|-?\d+)", rhs.split(" i64 ", 1)[1])
                ok = len(ops) >= 1 and all(is_narrow(op) for op in ops)
            elif rhs.startswith("select ") and " i64 " in rhs:
                ops = re.findall(r"(%[\w.]+|-?\d+)", rhs.split(" i64 ", 1)[1])
                ok = len(ops) >= 2 and all(is_narrow(op) for op in ops)
            elif re.match(r"sext i32 %[\w.]+ to i64$", rhs):
                ok = True
            if ok:
                narrow.add(name)
                changed = True

    if not narrow:
        return text

    fresh_n = 0

    def fresh():
        nonlocal fresh_n
        fresh_n += 1
        return f"%n64_{fresh_n}"

    out = []
    for line in lines:
        match = re.match(r"^(\s*)(%[\w.]+) = (.*\S)\s*$", line)
        if not match:
            out.append(line)
            continue
        indent, name, rhs = match.group(1), match.group(2), match.group(3)
        if name not in narrow:
            # An i8 GEP index moves 2 bytes on this BirenLLC, so an
            # 8-byte int64 element is index * 4. A plain getelementptr i64
            # moves 16 bytes and returns value 2*i.
            i64_gep = re.match(
                r"getelementptr i64, (ptr addrspace\(\d+\) %[\w.]+), i32 (%[\w.]+|-?\d+)$",
                rhs,
            )
            if i64_gep:
                ptr, idx = i64_gep.group(1), i64_gep.group(2)
                off = fresh()
                # BirenLLC treats this i8 index as a 2-byte step, so 4
                # here is an 8-byte int64 element.
                out.append(f"{indent}{off} = mul i32 {idx}, 4")
                out.append(
                    f"{indent}{name} = getelementptr i8, {ptr}, i32 {off}"
                )
                continue
            gep_idx = re.match(
                r"getelementptr ([\w.]+), (ptr addrspace\(\d+\) %[\w.]+), i64 (%[\w.]+)$",
                rhs,
            )
            if gep_idx and gep_idx.group(3) in narrow:
                ty, ptr, idx = gep_idx.group(1), gep_idx.group(2), gep_idx.group(3)
                out.append(
                    f"{indent}{name} = getelementptr {ty}, {ptr}, i32 {idx}"
                )
                continue
            out.append(line)
            continue
        if rhs.startswith("load i64,"):
            rhs = re.sub(
                r"^load i64, (ptr addrspace\(\d+\) %[\w.]+), align \d+$",
                r"load i32, \1, align 4",
                rhs,
            )
        elif rhs.startswith("phi i64 "):
            rhs = rhs.replace("phi i64 ", "phi i32 ", 1)
        elif " i64 " in rhs and rhs.startswith(
            ("add ", "sub ", "mul ", "shl ", "lshr ", "ashr ", "and ", "or ", "xor ", "icmp ")
        ):
            rhs = rhs.replace(" i64 ", " i32 ")
        elif rhs.startswith("select ") and " i64 " in rhs:
            rhs = rhs.replace(" i64 ", " i32 ")
        elif re.match(r"sext i32 (%[\w.]+) to i64$", rhs):
            src = re.match(r"sext i32 (%[\w.]+) to i64$", rhs).group(1)
            rhs = f"add i32 {src}, 0"
        out.append(f"{indent}{name} = {rhs}")
    text = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    # Uses such as ``sitofp i64 %v`` still name the pre-trunc type.
    text = re.sub(
        r"\bi64 (%[\w.]+)",
        lambda match: "i32 " + match.group(1)
        if match.group(1) in narrow
        else match.group(0),
        text,
    )
    text = re.sub(
        r"^(\s*)(%[\w.]+) = trunc i32 (%[\w.]+) to i32\s*$",
        r"\1\2 = add i32 \3, 0",
        text,
        flags=re.M,
    )
    return text


def _lower_biren_shared(text):
    """Put shared memory in Biren's addrspace(2).

    brcc lowers ``__shared__`` to an internal addrspace(2) global.
    Triton emits CUDA's external addrspace(3) ``@global_smem``. BirenLLC
    still emits an ELF, but ``sudrvModuleGetFunction`` then fails with
    "Not converted status 1" as soon as that symbol is referenced.
    """
    text = re.sub(
        r"@global_smem = external addrspace\(3\) global (\[[0-9]+ x i8\]), align ([0-9]+)",
        r"@global_smem = internal unnamed_addr addrspace(2) global \1 undef, align \2",
        text,
    )
    return text.replace("ptr addrspace(3)", "ptr addrspace(2)")


def _lower_llvm_exp(text):
    """Replace ``llvm.exp.f32``. BirenLLC aborts while expanding it.

    Range-reduce ``x = n*ln2 + f`` and use a degree-6 Taylor polynomial
    for ``exp(f)``. ``2^n`` is the float constructed from exponent bits.
    """
    if "@llvm.exp.f32" not in text:
        return text
    counter = 0

    def fresh():
        nonlocal counter
        counter += 1
        return f"%exp{counter}"

    def expand(match):
        indent, dst, src = match.group(1), match.group(2), match.group(3)
        t = [fresh() for _ in range(24)]
        lines = [
            f"{indent}{t[0]} = fcmp olt float {src}, -8.000000e+01",
            f"{indent}{t[1]} = select i1 {t[0]}, float -8.000000e+01, float {src}",
            f"{indent}{t[2]} = fcmp ogt float {t[1]}, 8.000000e+01",
            f"{indent}{t[3]} = select i1 {t[2]}, float 8.000000e+01, float {t[1]}",
            f"{indent}{t[4]} = fmul float {t[3]}, 0x3FF7154760000000",
            f"{indent}{t[5]} = fcmp olt float {t[4]}, 0.000000e+00",
            f"{indent}{t[6]} = select i1 {t[5]}, float -5.000000e-01, float 5.000000e-01",
            f"{indent}{t[7]} = fadd float {t[4]}, {t[6]}",
            f"{indent}{t[8]} = fptosi float {t[7]} to i32",
            f"{indent}{t[9]} = sitofp i32 {t[8]} to float",
            f"{indent}{t[10]} = fmul float {t[9]}, 0x3FE62E4300000000",
            f"{indent}{t[11]} = fsub float {t[3]}, {t[10]}",
            f"{indent}{t[12]} = fmul float {t[11]}, 0x3F56C16C20000000",
            f"{indent}{t[13]} = fadd float {t[12]}, 0x3F81111120000000",
            f"{indent}{t[14]} = fmul float {t[11]}, {t[13]}",
            f"{indent}{t[15]} = fadd float {t[14]}, 0x3FA5555560000000",
            f"{indent}{t[16]} = fmul float {t[11]}, {t[15]}",
            f"{indent}{t[17]} = fadd float {t[16]}, 0x3FC5555560000000",
            f"{indent}{t[18]} = fmul float {t[11]}, {t[17]}",
            f"{indent}{t[19]} = fadd float {t[18]}, 5.000000e-01",
            f"{indent}{t[20]} = fmul float {t[11]}, {t[19]}",
            f"{indent}{t[21]} = fadd float {t[20]}, 1.000000e+00",
            f"{indent}{t[22]} = fmul float {t[11]}, {t[21]}",
            f"{indent}{t[23]} = fadd float {t[22]}, 1.000000e+00",
            f"{indent}{dst} = fadd float {t[23]}, 0.000000e+00",
        ]
        # 2^n from the biased exponent, then scale the polynomial.
        scale_i = fresh()
        scale_b = fresh()
        scale = fresh()
        lines[-1] = f"{indent}{scale_i} = add i32 {t[8]}, 127"
        lines.extend([
            f"{indent}{scale_b} = shl i32 {scale_i}, 23",
            f"{indent}{scale} = bitcast i32 {scale_b} to float",
            f"{indent}{dst} = fmul float {t[23]}, {scale}",
        ])
        return "\n".join(lines)

    text = re.sub(
        r"^(\s*)(%[\w.]+) = call float @llvm\.exp\.f32\(float (%[\w.]+)\)\s*$",
        expand,
        text,
        flags=re.M,
    )
    text = re.sub(
        r"(?m)^declare float @llvm\.exp\.f32\(float\).*\n",
        "",
        text,
    )
    text = re.sub(
        r"(?m)^attributes #\d+ = \{ nocallback nofree nosync nounwind speculatable willreturn memory\(none\) \}\n",
        "",
        text,
    )
    return text


def _lower_llvm_sincos(text):
    """Replace ``llvm.sin.f32`` and ``llvm.cos.f32``.

    BirenLLC accepts the intrinsics, but the results are not usable:
    ``cos(0)`` comes back as about ``0.0078125`` instead of ``1``.
    Range-reduce by ``pi`` with a Cody-Waite split, then evaluate
    degree-11 Taylor polynomials on ``[-pi/2, pi/2]``. The residual is
    well under one bfloat16 ulp.
    """
    if "@llvm.sin.f32" not in text and "@llvm.cos.f32" not in text:
        return text
    counter = 0
    sin_poly = (
        "-0.1666666716337204",
        "0.008333333767950535",
        "-0.00019841270113829523",
        "2.7557314297155244e-06",
        "-2.5050759688838298e-08",
        "1.5896910177115586e-10",
    )
    cos_poly = (
        "-0.5",
        "0.0416666679084301",
        "-0.0013888889225199819",
        "2.4801587642286904e-05",
        "-2.7557314297155244e-07",
        "2.087572337217125e-09",
    )

    def fresh():
        nonlocal counter
        counter += 1
        return f"%sc{counter}"

    def horner(indent, lines, z, coeffs):
        acc = fresh()
        lines.append(f"{indent}{acc} = fadd float {coeffs[-1]}, 0.000000e+00")
        for coeff in reversed(coeffs[:-1]):
            prod = fresh()
            nxt = fresh()
            lines.append(f"{indent}{prod} = fmul float {acc}, {z}")
            lines.append(f"{indent}{nxt} = fadd float {prod}, {coeff}")
            acc = nxt
        return acc

    def expand(which):
        def repl(match):
            indent, dst, src = match.group(1), match.group(2), match.group(3)
            lines = []
            scaled = fresh()
            lines.append(
                f"{indent}{scaled} = fmul float {src}, 0.31830987334251404"
            )
            negative = fresh()
            lines.append(
                f"{indent}{negative} = fcmp olt float {scaled}, 0.000000e+00"
            )
            half = fresh()
            lines.append(
                f"{indent}{half} = select i1 {negative}, "
                "float -5.000000e-01, float 5.000000e-01"
            )
            biased = fresh()
            lines.append(f"{indent}{biased} = fadd float {scaled}, {half}")
            turns = fresh()
            lines.append(f"{indent}{turns} = fptosi float {biased} to i32")
            turns_f = fresh()
            lines.append(f"{indent}{turns_f} = sitofp i32 {turns} to float")
            high = fresh()
            lines.append(
                f"{indent}{high} = fmul float {turns_f}, 3.1415927410125732"
            )
            reduced_hi = fresh()
            lines.append(f"{indent}{reduced_hi} = fsub float {src}, {high}")
            low = fresh()
            lines.append(
                f"{indent}{low} = fmul float {turns_f}, -8.742277657347586e-08"
            )
            reduced = fresh()
            lines.append(
                f"{indent}{reduced} = fsub float {reduced_hi}, {low}"
            )
            square = fresh()
            lines.append(
                f"{indent}{square} = fmul float {reduced}, {reduced}"
            )
            sin_series = horner(indent, lines, square, sin_poly)
            cos_series = horner(indent, lines, square, cos_poly)
            sin_inner = fresh()
            lines.append(
                f"{indent}{sin_inner} = fmul float {square}, {sin_series}"
            )
            sin_scale = fresh()
            lines.append(
                f"{indent}{sin_scale} = fadd float {sin_inner}, 1.000000e+00"
            )
            sin_v = fresh()
            lines.append(
                f"{indent}{sin_v} = fmul float {reduced}, {sin_scale}"
            )
            cos_inner = fresh()
            lines.append(
                f"{indent}{cos_inner} = fmul float {square}, {cos_series}"
            )
            cos_v = fresh()
            lines.append(
                f"{indent}{cos_v} = fadd float {cos_inner}, 1.000000e+00"
            )
            odd_bit = fresh()
            lines.append(f"{indent}{odd_bit} = and i32 {turns}, 1")
            odd = fresh()
            lines.append(f"{indent}{odd} = icmp ne i32 {odd_bit}, 0")
            neg_sin = fresh()
            neg_cos = fresh()
            lines.append(
                f"{indent}{neg_sin} = fsub float -0.000000e+00, {sin_v}"
            )
            lines.append(
                f"{indent}{neg_cos} = fsub float -0.000000e+00, {cos_v}"
            )
            sin_out = fresh()
            cos_out = fresh()
            lines.append(
                f"{indent}{sin_out} = select i1 {odd}, float {neg_sin}, float {sin_v}"
            )
            lines.append(
                f"{indent}{cos_out} = select i1 {odd}, float {neg_cos}, float {cos_v}"
            )
            chosen = sin_out if which == "sin" else cos_out
            lines.append(f"{indent}{dst} = fadd float {chosen}, 0.000000e+00")
            return "\n".join(lines)

        return repl

    text = re.sub(
        r"^(\s*)(%[\w.]+) = call float @llvm\.sin\.f32\(float ([^)\s]+)\)\s*$",
        expand("sin"),
        text,
        flags=re.M,
    )
    text = re.sub(
        r"^(\s*)(%[\w.]+) = call float @llvm\.cos\.f32\(float ([^)\s]+)\)\s*$",
        expand("cos"),
        text,
        flags=re.M,
    )
    text = re.sub(
        r"(?m)^; Function Attrs: nocallback nofree nosync nounwind "
        r"speculatable willreturn memory\(none\)\n",
        "",
        text,
    )
    text = re.sub(
        r"(?m)^declare float @llvm\.(?:sin|cos)\.f32\(float\).*\n",
        "",
        text,
    )
    text = re.sub(
        r"(?m)^attributes #\d+ = \{ nocallback nofree nosync nounwind "
        r"speculatable willreturn memory\(none\) \}\n",
        "",
        text,
    )
    return text


def _attach_biren_target_attrs(text):
    """BirenLLC legalizes i64 using the br100 target feature set.

    brcc always emits ``target-cpu"="br100"``. Without it, i64 index math
    still assembles, but ``sudrvModuleGetFunction`` does not return.
    """
    if 'attributes #0 =' in text:
        return text
    text, count = re.subn(
        r"(define biren_supa_kernel void @[\w.]+\(.*\))\s*\{",
        r"\1 #0 {",
        text,
        count=1,
    )
    if count != 1:
        return text
    attrs = (
        'attributes #0 = { nounwind "target-cpu"="br100" '
        '"target-features"="+inst-br1xx-alu,+inst-br1xx-lsa,'
        '+inst-br1xx-scal,+inst-br1xx-tensor" }\n'
    )
    marker = "!llvm.module.flags"
    if marker in text:
        text = text.replace(marker, attrs + marker, 1)
    else:
        text += "\n" + attrs
    return text


def _lower_bfloat_loads(text):
    """Load BF16 as i16 and bitcast.

    ``load bfloat`` does not return the stored element. ``store bfloat``
    does, which is why an FP32→BF16 cast compares equal while a BF16
    gather does not. i16 has the same 2-byte width.
    """
    counter = 0

    def repl(match):
        nonlocal counter
        counter += 1
        indent, dst, ptr, align = match.groups()
        tmp = f"%bfld{counter}"
        return (
            f"{indent}{tmp} = load i16, {ptr}, align {align}\n"
            f"{indent}{dst} = bitcast i16 {tmp} to bfloat"
        )

    return re.sub(
        r"^(\s*)(%[\w.]+) = load bfloat, (ptr(?: addrspace\(\d+\))? %[\w.]+), align (\d+)\s*$",
        repl,
        text,
        flags=re.M,
    )


def biren_llvm_compat(text):
    """Drop LLVM syntax Biren's clang 16 parser does not accept.

    `or disjoint` is valid LLVM 18 IR and a no-op on the bitwise
    result. Vendor BirenLLC is the clang 16 toolchain from BRCC.
    """
    text = re.sub(r"\bor\s+disjoint\s+", "or ", text)
    text = _scalarize_singleton_structs(text)
    text = _lower_bfloat_casts(text)
    text = _lower_bfloat_loads(text)
    text = _narrow_i64_index_math(text)
    text = _lower_biren_shared(text)
    text = _lower_llvm_exp(text)
    text = _lower_llvm_sincos(text)
    text = _attach_biren_target_attrs(text)
    text = _renumber_numeric_values(text)
    return text


_DEFAULT_BIREN_SDK = (
    "/usr/local/birensupa/sdk/1.10.0.0-24145.lkg"
)
_BIREN_MCPU = {
    100: b"-mcpu=br100",
    104: b"-mcpu=br104",
    110: b"-mcpu=br110",
}
# BirenLLC keeps the first -mcpu for the process. Later compiles must match.
_LLC_LOCK = threading.Lock()
_LLC_CONFIG = None  # (arch, is_tmode)


class _BirenLinkElf(ctypes.Structure):
    _fields_ = [
        ("buffer", ctypes.c_void_p),
        ("size", ctypes.c_longlong),
    ]


def _biren_sdk():
    return Path(os.environ.get("BIREN_SDK", _DEFAULT_BIREN_SDK))


def _tool_path(env_name, default_rel):
    value = os.environ.get(env_name)
    path = Path(value).expanduser() if value else _biren_sdk() / default_rel
    if not path.is_file():
        raise RuntimeError(
            f"{env_name} does not exist: {path}. "
            "BirenLLC/BirenLink are libBirenLLC.so / libBirenLink.so, "
            "not standalone llc/ld binaries. "
            "Override with BIREN_LLC / BIREN_LINK if the SDK layout differs."
        )
    return path


def _llc_status(lib, status, where):
    if status == 0:
        return
    desc = lib.birenLLCGetStatusDesc(status)
    msg = desc.decode() if desc else "<unknown>"
    raise RuntimeError(f"{where} failed: status={status}, {msg}")


def _compile_biren_subin(ir_text, arch):
    """LLVM text -> ET_REL subin via libBirenLLC.

    brcc -### keeps codegen in-process. The callable surface that
    already accepts G-mode LLVM is birenLLCLLVMIRToSubin(). argv[0]
    must be "llc"; the only accepted extra flag is -mcpu=br1xx.
    """
    if arch not in _BIREN_MCPU:
        raise RuntimeError(
            f"unsupported SUPA arch {arch}; "
            f"BirenLLC accepts {sorted(_BIREN_MCPU)}"
        )

    lib_path = _tool_path("BIREN_LLC", "brcc/lib/libBirenLLC.so.1.0")
    lib = ctypes.CDLL(str(lib_path))
    lib.birenLLCGetVersion.restype = ctypes.c_char_p
    lib.birenLLCGetStatusDesc.argtypes = [ctypes.c_int]
    lib.birenLLCGetStatusDesc.restype = ctypes.c_char_p
    lib.birenLLCIsInitialized.restype = ctypes.c_bool
    lib.birenLLCInitialize.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.birenLLCInitialize.restype = ctypes.c_int
    lib.birenLLCLLVMIRToSubin.argtypes = [
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.birenLLCLLVMIRToSubin.restype = ctypes.c_int
    lib.birenLLCFinalize.argtypes = []
    lib.birenLLCFinalize.restype = ctypes.c_int

    version = lib.birenLLCGetVersion()
    print(
        "[SUPA] BirenLLC:",
        lib_path,
        "version",
        version.decode() if version else "<unknown>",
        f"arch={arch}",
        "mode=G",
        flush=True,
    )

    # LLVM command-line parsing inside birenLLCInitialize accepts
    # -mcpu only on the first call in the process. G-mode only for now.
    is_tmode = False
    config = (int(arch), is_tmode)
    with _LLC_LOCK:
        global _LLC_CONFIG
        if _LLC_CONFIG is None:
            argv = (ctypes.c_char_p * 2)(b"llc", _BIREN_MCPU[arch])
            _llc_status(
                lib, lib.birenLLCInitialize(2, argv), "birenLLCInitialize"
            )
            _LLC_CONFIG = config
        elif _LLC_CONFIG != config:
            raise RuntimeError(
                "BirenLLC already initialized with incompatible options "
                f"{_LLC_CONFIG}, requested {config}"
            )

    libc = ctypes.CDLL(None)
    libc.free.argtypes = [ctypes.c_void_p]
    src = ir_text.encode()
    out_ptr = ctypes.c_void_p()
    out_size = ctypes.c_size_t()
    _llc_status(
        lib,
        lib.birenLLCLLVMIRToSubin(
            arch,
            False,  # G-mode. T-mode is the tensor ISA, not this kernel.
            False,
            src,
            len(src),
            ctypes.byref(out_ptr),
            ctypes.byref(out_size),
        ),
        "birenLLCLLVMIRToSubin",
    )
    subin = ctypes.string_at(out_ptr, out_size.value)
    libc.free(out_ptr)
    print(
        f"[SUPA] BirenLLC SUCCESS subin={len(subin)} bytes",
        flush=True,
    )
    return subin


def _link_biren_elf(subin):
    """ET_REL subin -> ET_EXEC image via libBirenLink.biren_link."""
    lib_path = _tool_path("BIREN_LINK", "brcc/lib/libBirenLink.so.1.0")
    link = ctypes.CDLL(str(lib_path))
    link.biren_link.argtypes = [
        ctypes.c_char_p,
        ctypes.c_ulonglong,
        ctypes.POINTER(_BirenLinkElf),
    ]
    link.biren_link.restype = ctypes.c_int
    print("[SUPA] BirenLink:", lib_path, flush=True)

    linked = _BirenLinkElf()
    status = link.biren_link(subin, len(subin), ctypes.byref(linked))
    if status != 0:
        if linked.buffer:
            err = ctypes.string_at(linked.buffer).decode(errors="replace")
        else:
            err = "<unknown>"
        raise RuntimeError(
            f"biren_link failed: status={status}, error={err}"
        )

    image = ctypes.string_at(linked.buffer, linked.size)
    libc = ctypes.CDLL(None)
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free(linked.buffer)
    print(
        f"[SUPA] BirenLink SUCCESS birelf={len(image)} bytes",
        flush=True,
    )
    return image


DUMP_DIR = Path("/data/biren-oracle/triton-supa")
# Full IR dumps stay on by default. Set SUPA_DUMP_IR=0 to keep only files.
_DUMP_IR = os.environ.get("SUPA_DUMP_IR", "1") != "0"


def _emit(text):
    if _DUMP_IR:
        print(text, flush=True)


@dataclass(frozen=True)
class SUPAOptions:
    arch: int = 100

    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2

    warp_size: int = 32

    # Triton JIT automatically injects these.
    debug: bool = False
    instrumentation_mode: str = ""
    fpsan_homomorphic_casts: bool = False

    # Common backend metadata, matching CUDA/HIP Options.
    backend_name: str = "supa"
    sanitize_overflow: bool = True

    def hash(self):
        key = "_".join(
            f"{k}-{v}"
            for k, v in self.__dict__.items()
        )

        return hashlib.sha256(
            key.encode()
        ).hexdigest()


class SUPABackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "supa"

    def __init__(self, target: GPUTarget):
        super().__init__(target)

        # 最终输出是经过 BirenLink 的可加载 ELF
        self.binary_ext = "birelf"

    def hash(self):
        return (
            f"supa-"
            f"{self.target.arch}-"
            f"{self.target.warp_size}-v0"
        )

    def parse_options(self, opts):
        print(
            "[SUPA] parse_options:",
            opts,
            flush=True,
        )

        args = {
            "arch": int(self.target.arch),
            "warp_size": int(self.target.warp_size),
        }

        args.update({
            key: opts[key]
            for key in SUPAOptions.__dataclass_fields__
            if key in opts
            and opts[key] is not None
        })

        options = SUPAOptions(**args)

        print(
            "[SUPA] parsed options:",
            options,
            flush=True,
        )

        return options

    def pack_metadata(self, metadata):
        # Triton runtime passes this tuple to the generated
        # launcher as kernel_metadata. The C side parses "iii".
        return (
            metadata.num_warps,
            metadata.num_ctas,
            getattr(metadata, "shared", 0),
        )

    def get_codegen_implementation(self, options):
        # VectorAdd does not use tl.dot. Keep a stub so
        # ast_to_ttir can attach codegen_fns to the builder.
        return {
            "min_dot_size": lambda lhsType, rhsType: (1, 1, 1),
        }

    def load_dialects(self, context):
        # 第一阶段还没有 SUPA-specific MLIR dialect.
        pass

    def get_module_map(self):
        from triton.language.extra.supa import libdevice

        return {"triton.language.extra.libdevice": libdevice}

    @staticmethod
    def make_ttir(mod, metadata, options):
        print("[SUPA] make_ttir input:", flush=True)
        _emit(mod)

        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        (DUMP_DIR / "vector_add.before_ttir.mlir").write_text(
            str(mod)
        )

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # Generic TTIR cleanup only. No SUPA-specific passes.
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)

        pm.run(mod, "make_ttir")

        print("[SUPA] make_ttir output:", flush=True)
        _emit(mod)

        (DUMP_DIR / "vector_add.after_ttir.mlir").write_text(
            str(mod)
        )

        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        print("\n[SUPA] make_ttgir input:", flush=True)
        _emit(mod)

        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        (DUMP_DIR / "vector_add.before_ttgir.mlir").write_text(
            str(mod)
        )

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        #
        # TTIR -> TritonGPU IR.
        #
        # "supa:100" is only a TTGIR target annotation.
        # ConvertTritonToTritonGPU does not parse SUPA
        # architecture here; it records ttg.target /
        # num-warps / threads-per-warp / num-ctas and
        # assigns blocked layouts.
        #
        passes.ttir.add_convert_to_ttgpuir(
            pm,
            f"supa:{options.arch}",
            options.num_warps,
            options.warp_size,
            options.num_ctas,
        )

        # Generic TritonGPU transforms. No SUPA-specific
        # or matmul/descriptor passes in this MVP.
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)

        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        pm.run(mod, "make_ttgir")

        print("\n[SUPA] make_ttgir output:", flush=True)
        _emit(mod)

        (DUMP_DIR / "vector_add.ttgir.mlir").write_text(
            str(mod)
        )

        metadata["tensordesc_meta"] = (
            mod.get_tensordesc_metadata()
        )

        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        from triton._C.libtriton import llvm, supa

        mod = src
        print("\n[SUPA] make_llir input:", flush=True)
        _emit(mod)

        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        (DUMP_DIR / "vector_add.before_llir.mlir").write_text(str(mod))

        # Capture the Triton entry before tt.func is lowered away.
        kernel_name = ""
        if hasattr(mod, "get_entry_func_name"):
            kernel_name = mod.get_entry_func_name()
        if not kernel_name:
            raise RuntimeError("Cannot determine SUPA kernel entry")
        metadata["name"] = kernel_name

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        # tt.reduce across warps reads allocation.offset from this pass.
        passes.ttgpuir.add_allocate_shared_memory(pm)
        supa.passes.ttgpuir.add_to_llvmir(pm, options.arch)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.convert.add_arith_to_llvmir(pm)

        pm.run(mod, "make_llir")

        print("\n===== LLVM DIALECT =====", flush=True)
        _emit(mod)
        (DUMP_DIR / "vector_add.llir.mlir").write_text(str(mod))

        llvm.init_targets()
        ctx = llvm.context()
        llvm_mod = llvm.to_module(mod, ctx)
        llir = str(llvm_mod)
        del llvm_mod
        del ctx

        print("\n===== LLVM IR (pre-finalize) =====", flush=True)
        _emit(llir)
        (DUMP_DIR / "vector_add.ll").write_text(llir)

        llir = finalize_supa_llvm(llir, kernel_name, metadata, options)

        print("\n===== BIREN LLVM IR =====", flush=True)
        _emit(llir)
        (DUMP_DIR / "vector_add.biren.ll").write_text(llir)

        shared = None
        if hasattr(mod, "get_int_attr"):
            shared = mod.get_int_attr("ttg.shared")
        metadata["shared"] = shared or 0
        return llir

    @staticmethod
    def make_birelf(src, metadata, options):
        if not isinstance(src, str):
            raise TypeError("Expected Biren LLVM IR as string")

        ir_text = biren_llvm_compat(src)
        tag = (
            f"{metadata.get('name', 'kernel')}"
            f"_w{options.num_warps}"
        )
        print(
            "\n===== BIREN LLVM IR (POST COMPAT / LLC INPUT) =====",
            flush=True,
        )
        _emit(ir_text)
        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        (DUMP_DIR / f"{tag}.post_compat.ll").write_text(ir_text)

        tmp_dir = tempfile.mkdtemp(prefix="triton-supa-")
        tmp = Path(tmp_dir)
        keep_artifacts = True
        try:
            ll_file = tmp / "kernel.ll"
            subin_file = tmp / "kernel.subin"
            elf_file = tmp / "kernel.birelf"
            ll_file.write_text(ir_text)
            (DUMP_DIR / "vector_add.biren.llc.ll").write_text(ir_text)
            (DUMP_DIR / f"{tag}.biren.llc.ll").write_text(ir_text)

            subin = _compile_biren_subin(ir_text, int(options.arch))
            subin_file.write_bytes(subin)
            (DUMP_DIR / "vector_add.subin").write_bytes(subin)

            image = _link_biren_elf(subin)
            elf_file.write_bytes(image)
            (DUMP_DIR / "vector_add.birelf").write_bytes(image)
            (DUMP_DIR / f"{tag}.birelf").write_bytes(image)

            keep_artifacts = False
            return image
        finally:
            if not keep_artifacts:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                print(
                    f"[SUPA] Biren codegen artifacts kept in {tmp}",
                    flush=True,
                )

    def add_stages(
        self,
        stages,
        options,
        language=Language.TRITON,
    ):
        print(
            "[SUPA] add_stages reached",
            options,
            flush=True,
        )

        if language != Language.TRITON:
            raise RuntimeError(
                "SUPA backend currently supports "
                "Triton input only."
            )

        stages["ttir"] = (
            lambda src, metadata: self.make_ttir(
                src,
                metadata,
                options,
            )
        )
        stages["ttgir"] = (
            lambda src, metadata: self.make_ttgir(
                src,
                metadata,
                options,
            )
        )
        stages["llir"] = (
            lambda src, metadata: self.make_llir(
                src,
                metadata,
                options,
            )
        )
        stages["birelf"] = (
            lambda src, metadata: self.make_birelf(
                src,
                metadata,
                options,
            )
        )
