#!/usr/bin/env python3
"""rope_cast_probe.py — read-only: does a .litertlm bundle compute its rotary angles inside the graph from
an INT32 position cast to float?  That is the structure behind LiteRT-LM #3913 (PowerBeef 2026-10-13:
input_pos INT32 -> CAST -> MUL -> SIN / COS; a float16 GPU path cannot separate integers above 2048).

For every TF_LITE section and every signature it prints: the op count, the SIN / COS op count, the
INT32 -> float CAST ops, and which of those casts reach a SIN or COS op within MAX_HOPS consumers (the
cast's input tensor name and the op chain).  It also lists the signature's INT32 inputs with the ops
that read them (CAST = computed in the graph, GATHER = table lookup).  Nothing is written; the file is
mmapped and read through the lazy flatbuffer accessors, so a multi-GB bundle costs seconds.

usage: python3 -I rope_cast_probe.py <bundle.litertlm>...
exit 0 always; the verdict line per bundle is `ROTARY-CAST-IN-GRAPH <yes|no> <path>`.
"""
import mmap
import os
import struct
import sys
from collections import deque

from ai_edge_litert import schema_py_generated as S
from litert_lm_builder import litertlm_header_schema_py_generated as H

MAX_HOPS = 8
OP_NAME = {v: k for k, v in vars(S.BuiltinOperator).items() if isinstance(v, int)}
TYPE_NAME = {v: k for k, v in vars(S.TensorType).items() if isinstance(v, int)}
FLOATS = (S.TensorType.FLOAT32, S.TensorType.FLOAT16)


def read_header(path):
    """(header_end, [(begin, end, model_type)]) — the same walk as PowerBeef's patch_rope.py."""
    with open(path, "rb") as f:
        header_end = struct.unpack("<Q", f.read(32)[24:32])[0]
        f.seek(0)
        header = f.read(header_end)
    md = H.LiteRTLMMetaData.GetRootAs(header, 32)
    sections = []
    for i in range(md.SectionMetadata().ObjectsLength()):
        o = md.SectionMetadata().Objects(i)
        kind = None
        for j in range(o.ItemsLength()):
            kv = o.Items(j)
            if kv.Key().decode() == "model_type":
                v = H.StringValue()
                v.Init(kv.Value().Bytes, kv.Value().Pos)
                kind = v.Value().decode()
        sections.append((o.BeginOffset(), o.EndOffset(), kind))
    return header_end, sections


def probe_subgraph(model, sg, sig):
    nops = sg.OperatorsLength()
    codes, ins, outs = [], [], []
    for j in range(nops):
        op = sg.Operators(j)
        oc = model.OperatorCodes(op.OpcodeIndex())
        codes.append(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
        ins.append([int(op.Inputs(k)) for k in range(op.InputsLength())])
        outs.append([int(op.Outputs(k)) for k in range(op.OutputsLength())])
    consumers = {}
    for j in range(nops):
        for t in ins[j]:
            if t >= 0:
                consumers.setdefault(t, []).append(j)

    def ttype(t):
        return sg.Tensors(t).Type()

    def tname(t):
        n = sg.Tensors(t).Name()
        return n.decode(errors="replace") if n else f"#{t}"

    trig = [j for j in range(nops) if codes[j] in (S.BuiltinOperator.SIN, S.BuiltinOperator.COS)]
    casts = [j for j in range(nops)
             if codes[j] == S.BuiltinOperator.CAST and ins[j] and ins[j][0] >= 0 and outs[j]
             and ttype(ins[j][0]) == S.TensorType.INT32 and ttype(outs[j][0]) in FLOATS]
    rotary = []
    for j in casts:
        q, seen, hit = deque([(j, [j])]), {j}, None
        while q and hit is None:
            cur, path = q.popleft()
            if len(path) > MAX_HOPS:
                continue
            for t in outs[cur]:
                for nxt in consumers.get(t, []):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    p2 = path + [nxt]
                    if codes[nxt] in (S.BuiltinOperator.SIN, S.BuiltinOperator.COS):
                        hit = p2
                        break
                    q.append((nxt, p2))
                if hit:
                    break
        if hit:
            rotary.append((tname(ins[j][0]), tname(outs[j][0]), TYPE_NAME[ttype(outs[j][0])],
                           " > ".join(OP_NAME.get(codes[k], str(codes[k])) for k in hit)))

    int_inputs = []
    for k in range(sig.InputsLength()):
        ti = sig.Inputs(k).TensorIndex()
        if ttype(ti) == S.TensorType.INT32:
            readers = sorted({OP_NAME.get(codes[c], str(codes[c])) for c in consumers.get(ti, [])})
            int_inputs.append((sig.Inputs(k).Name().decode(), readers))
    return nops, len(trig), len(casts), rotary, int_inputs


def probe(path):
    header_end, sections = read_header(path)
    verdict = False
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for begin, end, kind in sections:
            if not (kind or "").startswith("tf_lite"):
                continue
            model = S.Model.GetRootAs(mm, begin)
            for i in range(model.SignatureDefsLength()):
                sig = model.SignatureDefs(i)
                sg = model.Subgraphs(sig.SubgraphIndex())
                nops, ntrig, ncast, rotary, int_inputs = probe_subgraph(model, sg, sig)
                verdict = verdict or bool(rotary)
                print(f"  [{kind}] {sig.SignatureKey().decode()}: ops {nops}, SIN/COS {ntrig}, "
                      f"INT32->float CAST {ncast}, cast reaching SIN/COS {len(rotary)}")
                for cin, cout, ctype, chain in rotary:
                    print(f"      {cin} -> {cout} ({ctype}): {chain}")
                for name, readers in int_inputs:
                    print(f"      INT32 input {name}: read by {', '.join(readers) or '(nothing)'}")
    print(f"ROTARY-CAST-IN-GRAPH {'yes' if verdict else 'no'} {path}")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    for p in argv[1:]:
        print(f"== {os.path.basename(p)} ({os.path.getsize(p):,} bytes)")
        try:
            probe(p)
        except Exception as e:  # keep going; the verdict line says what failed
            print(f"ROTARY-CAST-IN-GRAPH error {p}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
