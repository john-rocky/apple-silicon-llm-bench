#!/usr/bin/env python3
"""composite_count.py — read-only: how does each text signature of a .litertlm bundle compute its rotary angles?

For every TF_LITE section and every signature it prints the op count, the STABLEHLO_COMPOSITE ops grouped by
composite name (with the decoded attributes of the first `odml.rope` and the op count of its decomposition
subgraph), the SIN / COS op count, and the INT32 -> float CAST ops that reach a SIN or COS within MAX_HOPS
consumers (the LiteRT-LM #3913 chain: input_pos -> CAST -> MUL -> SIN / COS).  Companion of rope_cast_probe.py
(same section walk, same chain search); nothing is written, the file is mmapped.

usage: python3 -I composite_count.py <bundle.litertlm>...
verdict line per bundle (main graphs only, decomposition subgraphs excluded):
  ROPE-LOWERING <composite|cast-chain|both|neither> <path>
"""
import mmap
import os
import struct
import sys
from collections import Counter, deque

from ai_edge_litert import schema_py_generated as S
from litert_lm_builder import litertlm_header_schema_py_generated as H

try:
    from flatbuffers import flexbuffers
except ImportError:  # the attributes are then printed as raw bytes
    flexbuffers = None

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


def composite_info(op):
    """(name, decomposition subgraph index, attributes) of a STABLEHLO_COMPOSITE op, or None."""
    if op.BuiltinOptions2Type() != S.BuiltinOptions2.StableHLOCompositeOptions:
        return None
    t = op.BuiltinOptions2()
    opts = S.StableHLOCompositeOptions()
    opts.Init(t.Bytes, t.Pos)
    name = opts.Name().decode() if opts.Name() else "?"
    attrs = None
    if opts.CompositeAttributesLength():
        raw = bytes(opts.CompositeAttributesAsNumpy())
        if flexbuffers is not None:
            try:
                attrs = flexbuffers.Loads(raw)
            except Exception as e:  # noqa: BLE001
                attrs = f"<flexbuffer {len(raw)} B, undecodable: {type(e).__name__}>"
        else:
            attrs = f"<flexbuffer {len(raw)} B>"
    return name, opts.DecompositionSubgraphIndex(), attrs


def analyse(model, sg):
    nops = sg.OperatorsLength()
    codes, ins, outs, comps = [], [], [], []
    for j in range(nops):
        op = sg.Operators(j)
        oc = model.OperatorCodes(op.OpcodeIndex())
        codes.append(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
        ins.append([int(op.Inputs(k)) for k in range(op.InputsLength())])
        outs.append([int(op.Outputs(k)) for k in range(op.OutputsLength())])
        if codes[-1] == S.BuiltinOperator.STABLEHLO_COMPOSITE:
            comps.append(composite_info(op))
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

    trig = sum(1 for j in range(nops) if codes[j] in (S.BuiltinOperator.SIN, S.BuiltinOperator.COS))
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
            rotary.append((tname(ins[j][0]), " > ".join(OP_NAME.get(codes[k], str(codes[k])) for k in hit)))
    return nops, trig, len(casts), rotary, comps


def probe(path):
    header_end, sections = read_header(path)
    has_comp = has_chain = False
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for begin, end, kind in sections:
            if not (kind or "").startswith("tf_lite"):
                continue
            model = S.Model.GetRootAs(mm, begin)
            sig_subgraphs = {model.SignatureDefs(i).SubgraphIndex(): model.SignatureDefs(i).SignatureKey().decode()
                             for i in range(model.SignatureDefsLength())}
            print(f"  [{kind}] {model.SubgraphsLength()} subgraphs, {len(sig_subgraphs)} signatures")
            for i in range(model.SignatureDefsLength()):
                sig = model.SignatureDefs(i)
                sg = model.Subgraphs(sig.SubgraphIndex())
                nops, ntrig, ncast, rotary, comps = analyse(model, sg)
                names = Counter(c[0] for c in comps if c)
                print(f"    {sig.SignatureKey().decode()}: ops {nops}, composites {sum(names.values())}"
                      f" {{{', '.join(f'{k}: {v}' for k, v in sorted(names.items()))}}}, SIN/COS {ntrig},"
                      f" INT32->float CAST {ncast}, cast reaching SIN/COS {len(rotary)}")
                for cin, chain in rotary:
                    print(f"      cast-chain from {cin}: {chain}")
                    has_chain = True
                for cname in sorted({c[0] for c in comps if c and c[0] in ("odml.rope", "odml.qkv_norm_rope")}):
                    dist = Counter(repr(c[2]) for c in comps if c and c[0] == cname)
                    print(f"      {cname} attrs: " + "; ".join(f"{n} x {a}" for a, n in sorted(dist.items())))
                shown = set()
                for c in comps:
                    if c and c[0] in ("odml.rope", "odml.qkv_norm_rope") and c[0] not in shown:
                        shown.add(c[0])
                        has_comp = True
                        dsg = model.Subgraphs(c[1]) if 0 <= c[1] < model.SubgraphsLength() else None
                        dops = dsg.OperatorsLength() if dsg else "?"
                        dname = dsg.Name().decode(errors="replace") if dsg and dsg.Name() else ""
                        dtrig = analyse(model, dsg)[1] if dsg else "?"
                        print(f"      {c[0]}: attrs {c[2]}; decomposition subgraph {c[1]} ({dname}) ops {dops}, SIN/COS {dtrig}")
    verdict = {(True, True): "both", (True, False): "composite", (False, True): "cast-chain", (False, False): "neither"}[(has_comp, has_chain)]
    print(f"ROPE-LOWERING {verdict} {path}")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    for p in argv[1:]:
        print(f"== {os.path.basename(p)} ({os.path.getsize(p):,} bytes)")
        try:
            probe(p)
        except Exception as e:  # keep going; the verdict line says what failed
            print(f"ROPE-LOWERING error {p}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
