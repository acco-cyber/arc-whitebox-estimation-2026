import sys
import pyarrow.parquet as pq

p = sys.argv[1]
pf = pq.ParquetFile(p)
print("rows:", pf.metadata.num_rows, "row_groups:", pf.metadata.num_row_groups)
print(pf.schema_arrow)
for i in range(pf.metadata.num_row_groups):
    rg = pf.metadata.row_group(i)
    print("rg", i, "rows", rg.num_rows, "bytes", rg.total_byte_size)
# small columns of every row
small = [n for n in pf.schema_arrow.names if n not in ("weights",)]
print("small cols:", small)
t = pf.read(columns=[c for c in small if c not in ("final_means", "all_layer_means", "means", "layer_means")])
print(t.slice(0, 4).to_pylist())
