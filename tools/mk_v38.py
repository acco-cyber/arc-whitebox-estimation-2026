import re
p = r"E:\Claude code\whest\est\v38.py"
s = open(p, encoding="utf-8").read()
a = '''                R1T_st = r1b[:k_b + 1]
                R2T_st = r2b[:k_b + 1]
            w2b_list.append(w2)'''
b = '''                R1T_st = r1b[:k_b + 1]
                R2T_st = r2b[:k_b + 1]
            if HUB_A > 0.0 and riders:
                # V38: hub columns of units whose ReLU is (almost) linear or (almost) dead create
                # no new third cumulant: drop them from the newborn source (A columns, P columns
                # at the next layer, the hub rows of the static thin factors)
                hub_mask = (fnp.abs(alpha) < HUB_A).astype(f32)
                fnp.multiply(a_b, hub_mask[None, :], out=a_b)
                fnp.multiply(lb[k_b], hub_mask[:, None], out=lb[k_b])
                fnp.multiply(Rr_full, hub_mask[:, None], out=Rr_full)
                if rfb > 0:
                    fnp.multiply(r1b[k_b], hub_mask[:, None], out=r1b[k_b])
                    fnp.multiply(r2b[k_b], hub_mask[:, None], out=r2b[k_b])
            w2b_list.append(w2)'''
assert s.count(a) == 1
s = s.replace(a, b)
a2 = '''                fnp.copyto(legs["AP0"][k, 1], W)
'''
b2 = '''                fnp.copyto(legs["AP0"][k, 1], W)
                if hub_mask is not None:
                    fnp.multiply(legs["AP0"][k, 1], hub_mask[None, :], out=legs["AP0"][k, 1])
                    hub_mask = None
'''
assert s.count(a2) == 1
s = s.replace(a2, b2)
a3 = '''        perm_prev = None
        cdiag_prev = None'''
b3 = '''        perm_prev = None
        hub_mask = None   # V38: hub-column mask of the source born at the previous layer
        cdiag_prev = None'''
assert s.count(a3) == 1
s = s.replace(a3, b3)
a4 = '''NOJOIN_FROM = int(_os.environ.get("V36_NOJOIN", "99"))'''
b4 = a4 + '''
# V38: |alpha| bound of the hub columns a newborn source keeps (0 = all)
HUB_A = float(_os.environ.get("V38_HUB_A", "0"))'''
assert s.count(a4) == 1
s = s.replace(a4, b4)
open(p, "w", encoding="utf-8", newline="\n").write(s)
import ast; ast.parse(s); print("ok")
