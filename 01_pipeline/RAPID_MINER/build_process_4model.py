#!/usr/bin/env python3
"""Bangun process RapidMiner 4 model (RF, LR, XGBoost, MLP) untuk PD_MDIK.

Output default: 01_pipeline/RAPID_MINER/proyek_4model.rmp
Process lama `proyek_akhir.rmp` TIDAK disentuh (hanya dibaca sebagai acuan format).

Struktur process: preprocess sama dengan proyek_akhir.rmp (Read CSV -> Select
Attributes -> Nominal to Numerical -> Numerical to Polynominal -> Set Role ->
Split Data stratified 80/20 seed 42), lalu empat cabang learner yang semuanya
memakai partisi training dan partisi test yang sama:

    Split Data ─┬─> Random Forest      -> Apply Model -> Write CSV -> Performance -> result 1
                ├─> Logistic Regression-> Apply Model -> Write CSV -> Performance -> result 2
                ├─> XGBoost            -> Apply Model -> Write CSV -> Performance -> result 3
                └─> Neural Net (MLP)   -> Apply Model -> Write CSV -> Performance -> result 4

Catatan penting:
- XGBoost BUKAN operator bawaan AI Studio; butuh extension "XGBoost"
  (marketplace id rmx_xgboost, operator key `xgboost:xgboost`). Tanpa extension
  process gagal dimuat. Pakai `--tanpa-xgb` untuk versi 3 model.
- Logistic Regression RapidMiner = kernel logistic regression (kernel dot =
  linear) dan tidak punya parameter class_weight; MLP RapidMiner memakai
  sigmoid + backprop (bukan Adam/early stopping seperti scikit-learn).
  Perbedaan ini didokumentasikan di README, bukan dianggap identik.

Pemakaian:
  python 01_pipeline/RAPID_MINER/build_process_4model.py                # 4 model
  python 01_pipeline/RAPID_MINER/build_process_4model.py --tanpa-xgb    # 3 model
  python 01_pipeline/RAPID_MINER/build_process_4model.py --csv <path> \
      --out <path.rmp> --pred-dir <dir Windows> --models rf,lr,mlp      # uji
"""
from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
RM_DIR = Path(__file__).resolve().parent

DEFAULT_CSV = (r"RAPIDMINER_LOCAL_REPO"
               r"\_AO\PROYEK_PD_MDIK\modeling_dataset.csv")
DEFAULT_PRED_DIR = (r"RAPIDMINER_LOCAL_REPO"
                    r"\_AO\PROYEK_PD_MDIK")
VERSION = "12.1.001"

# Label harus BINOMINAL: Logistic Regression (SVM) menolak label polynominal
# ("Logistic Regression (SVM) cannot handle polynominal label") pada validasi
# design-time. RF/XGBoost/Neural Net menerima binominal, jadi satu konversi
# binominal dipakai untuk keempat cabang.
LABEL_OP = "Numerical to Binominal"

# 26 kolom dataset, urutan sama dengan modeling_dataset.csv
META = [
    ("time", "polynominal"), ("year", "integer"), ("month", "integer"),
    ("day", "integer"), ("latitude", "real"), ("longitude", "real"),
    ("mag", "real"), ("depth", "real"), ("nst", "real"), ("gap", "real"),
    ("dmin", "real"), ("rms", "real"), ("magType", "polynominal"),
    ("place", "polynominal"), ("label", "integer"),
    ("noaa_country", "polynominal"), ("noaa_location", "polynominal"),
    ("noaa_cause_code", "real"), ("noaa_water_height", "real"),
    ("noaa_intensity", "real"), ("depth_mag_ratio", "real"),
    ("abs_latitude", "real"), ("tectonic_zone", "polynominal"),
    ("monitoring_quality", "real"), ("era", "polynominal"),
    ("season", "polynominal"),
]

SELECTED = ("mag|depth|nst|gap|dmin|rms|latitude|longitude|abs_latitude|"
            "depth_mag_ratio|monitoring_quality|tectonic_zone|era|season|label")

MODELS: dict[str, dict] = {
    "rf": dict(
        name="Random Forest", cls="concurrency:parallel_random_forest",
        row=100, compat=VERSION, params=[
            ("number_of_trees", "200"), ("criterion", "gini_index"),
            ("maximal_depth", "15"), ("apply_pruning", "false"),
            ("confidence", "0.1"), ("apply_prepruning", "false"),
            ("minimal_gain", "0.0"), ("minimal_leaf_size", "1"),
            ("minimal_size_for_split", "2"),
            ("number_of_prepruning_alternatives", "3"),
            ("random_splits", "false"), ("guess_subset_ratio", "true"),
            ("subset_ratio", "0.2"), ("voting_strategy", "confidence vote"),
            ("use_local_random_seed", "true"), ("local_random_seed", "42"),
            ("enable_parallel_execution", "true"),
        ], extra="",
    ),
    "lr": dict(
        name="Logistic Regression", cls="logistic_regression",
        row=230, compat=VERSION, params=[
            ("kernel_type", "dot"), ("kernel_gamma", "1.0"),
            ("kernel_sigma1", "1.0"), ("kernel_sigma2", "0.0"),
            ("kernel_sigma3", "2.0"), ("kernel_shift", "1.0"),
            ("kernel_degree", "2.0"), ("kernel_a", "1.0"), ("kernel_b", "0.0"),
            ("kernel_cache", "200"), ("C", "1.0"),
            ("convergence_epsilon", "0.0010"),
            ("max_iterations", "100000"), ("scale", "true"),
            # Tidak ada calculate_weights / return_optimization_performance di
            # operator logistic_regression 12.1.1 (itu milik versi lain) —
            # menambahkannya memunculkan warning "parameter unknown" di GUI.
        ], extra="",
    ),
    "xgb": dict(
        name="XGBoost", cls="xgboost:xgboost", row=360,
        compat="0.1.000-SNAPSHOT", params=[
            ("booster", "tree booster"), ("rounds", "200"),
            ("early_stopping", "none"), ("early_stopping_rounds", "10"),
            ("learning_rate", "0.1"), ("min_split_loss", "0.0"),
            ("max_depth", "6"), ("min_child_weight", "1.0"),
            ("subsample", "1.0"), ("tree_method", "auto"),
            ("lambda", "1.0"), ("alpha", "0.0"),
            ("sample_type", "uniform"), ("normalize_type", "tree"),
            ("rate_drop", "0.0"), ("skip_drop", "0.0"),
            ("updater", "shotgun"), ("feature_selector", "cyclic"),
            ("top_k", "0"),
        ], extra='        <enumeration key="expert_parameters"/>\n',
    ),
    "mlp": dict(
        name="Neural Net MLP", cls="neural_net", row=490, compat=VERSION,
        params=[
            ("training_cycles", "200"), ("learning_rate", "0.3"),
            ("momentum", "0.2"), ("decay", "false"), ("shuffle", "true"),
            ("normalize", "true"), ("error_epsilon", "0.0"),
            ("use_local_random_seed", "true"), ("local_random_seed", "42"),
        ],
        # 32-16 (bukan 128-64-32 seperti Python): RM Neural Net memakai backprop
        # sigmoid tanpa Adam/early stopping, biaya per epoch sebanding jumlah bobot.
        # Dengan 128-64-32 batch 310k baris butuh berjam-jam.
        extra=('        <list key="hidden_layers">\n'
               '          <parameter key="Layer 1" value="32"/>\n'
               '          <parameter key="Layer 2" value="16"/>\n'
               '        </list>\n'),
    ),
}


def op_open(name: str, cls: str, x: int, y: int, compat: str,
            height: int = 82) -> str:
    return (f'      <operator activated="true" class="{cls}" '
            f'compatibility="{compat}" expanded="true" height="{height}" '
            f'name="{escape(name)}" width="90" x="{x}" y="{y}">\n')


def param(key: str, value: str) -> str:
    safe = escape(str(value), {'"': "&quot;"})
    return f'        <parameter key="{key}" value="{safe}"/>\n' 


def read_csv_block(csv_path: str) -> str:
    s = op_open("Read CSV", "read_csv", 45, 289, VERSION, 68)
    s += param("csv_file", csv_path)
    for k, v in [("column_separators", ","), ("trim_lines", "false"),
                 ("multiline_text", "false"), ("use_quotes", "true"),
                 ("quotes_character", '"'), ("escape_character", "\\"),
                 ("skip_comments", "true"), ("comment_characters", "#"),
                 ("starting_row", "1"), ("parse_numbers", "true"),
                 ("decimal_character", "."), ("grouped_digits", "false"),
                 ("grouping_character", ","), ("infinity_representation", ""),
                 ("date_format", "yyyy-MM-dd"), ("use_header_row", "true"),
                 ("header_row", "1"), ("first_row_as_names", "true")]:
        s += param(k, v)
    s += '        <list key="annotations"/>\n'
    for k, v in [("time_zone", "Asia/Makassar"),
                 ("locale", "English (United States)"),
                 ("encoding", "UTF-8"),
                 ("read_all_values_as_polynominal", "false")]:
        s += param(k, v)
    s += '        <list key="data_set_meta_data_information">\n'
    for i, (col, typ) in enumerate(META):
        s += f'          <parameter key="{i}" value="{col}.true.{typ}.attribute"/>\n'
    s += '        </list>\n'
    s += param("read_not_matching_values_as_missings", "true")
    s += '      </operator>\n'
    return s


def preprocess_blocks() -> str:
    s = op_open("Select Attributes", "select_attributes", 179, 289, VERSION)
    for k, v in [("attribute_filter_type", "subset"), ("attribute", ""),
                 ("attributes", SELECTED),
                 ("use_except_expression", "false"),
                 ("value_type", "attribute_value"),
                 ("use_value_type_exception", "false"),
                 ("except_value_type", "time"),
                 ("block_type", "attribute_block"),
                 ("use_block_type_exception", "false"),
                 ("except_block_type", "value_matrix_row_start"),
                 ("invert_selection", "false"),
                 ("include_special_attributes", "false")]:
        s += param(k, v)
    s += '      </operator>\n'

    s += op_open("Nominal to Numerical", "nominal_to_numerical", 313, 289,
                 VERSION, 103)
    for k, v in [("return_preprocessing_model", "false"),
                 ("attribute_filter_type", "subset"), ("attribute", ""),
                 ("attributes", "|tectonic_zone|era|season"),
                 ("use_except_expression", "false"), ("value_type", "nominal"),
                 ("use_value_type_exception", "false"),
                 ("except_value_type", "file_path"),
                 ("block_type", "single_value"),
                 ("use_block_type_exception", "false"),
                 ("except_block_type", "single_value"),
                 ("invert_selection", "false"),
                 ("include_special_attributes", "false"),
                 ("coding_type", "dummy coding"),
                 ("use_comparison_groups", "false"),
                 ("unexpected_value_handling", "all 0"),
                 ("use_underscore_in_name", "true")]:
        s += param(k, v)
    s += '        <list key="comparison_groups"/>\n'
    s += '      </operator>\n'

    s += op_open(LABEL_OP, "numerical_to_binominal", 447, 289, VERSION)
    for k, v in [("attribute_filter_type", "single"), ("attribute", "label"),
                 ("attributes", ""), ("use_except_expression", "false"),
                 ("value_type", "numeric"),
                 ("use_value_type_exception", "false"),
                 ("except_value_type", "real"),
                 ("block_type", "value_series"),
                 ("use_block_type_exception", "false"),
                 ("except_block_type", "value_series_end"),
                 ("invert_selection", "false"),
                 ("include_special_attributes", "false")]:
        s += param(k, v)
    s += '      </operator>\n'

    s += op_open("Set Role", "blending:set_role", 581, 289, VERSION)
    s += '        <list key="set_roles">\n'
    s += '          <parameter key="label" value="label"/>\n'
    s += '        </list>\n'
    s += '      </operator>\n'

    s += op_open("Split Data", "blending:split_data", 715, 289, VERSION, 103)
    s += '        <enumeration key="partitions">\n'
    s += '          <parameter key="ratio" value="0.8"/>\n'
    s += '          <parameter key="ratio" value="0.2"/>\n'
    s += '        </enumeration>\n'
    for k, v in [("sampling_type", "stratified sampling"),
                 ("use_local_random_seed", "true"),
                 ("local_random_seed", "42")]:
        s += param(k, v)
    s += '      </operator>\n'
    return s


def mlp_balance_block(y: int, neg_ratio: float) -> str:
    """Sample penyeimbang khusus cabang MLP.

    Neural Net RapidMiner tidak punya parameter class weight; pada 1:406 model
    kolaps ke "selalu negatif" (confidence konstan 0,0025). Cabang MLP karena
    itu dilatih pada data yang di-undersample: semua positif dipertahankan,
    negatif diambil sebagian (rasio 0,1 ~ 40:1). RF/LR/XGBoost tetap memakai
    data penuh. Rasio dipilih dari dua uji penuh 13 Nov 2026:
    0,025 (10:1) -> F1 0,137; 0,1 (40:1) -> F1 0,331.
    """
    s = op_open("Sample (MLP Balance)", "sample", 740, MODELS["mlp"]["row"],
                VERSION, 103)
    s += param("sample", "relative")
    s += param("balance_data", "true")
    s += param("sample_ratio", "1.0")
    s += '        <list key="sample_ratio_per_class">\n'
    s += f'          <parameter key="false" value="{neg_ratio}"/>\n'
    s += '          <parameter key="true" value="1.0"/>\n'
    s += '        </list>\n'
    for k, v in [("use_local_random_seed", "true"), ("local_random_seed", "42")]:
        s += param(k, v)
    s += '      </operator>\n'
    return s


def learner_block(key: str, y: int) -> str:
    spec = MODELS[key]
    s = op_open(spec["name"], spec["cls"], 849, y, spec["compat"],
                height=103)
    if spec["extra"].startswith("        <list"):
        s += spec["extra"]
    for k, v in spec["params"]:
        s += param(k, v)
    if spec["extra"].startswith("        <enumeration"):
        s += spec["extra"]
    s += '      </operator>\n'
    return s


def branch_blocks(key: str, y: int, pred_file: str, result_no: int) -> str:
    spec = MODELS[key]
    label = {k: MODELS[k]["name"] for k in MODELS}[key]
    s = op_open(f"Apply Model ({label})", "apply_model", 983, y, VERSION)
    s += '        <list key="application_parameters"/>\n'
    s += '      </operator>\n'

    s += op_open(f"Write CSV ({label})", "write_csv", 1117, y, VERSION)
    for k, v in [("csv_file", pred_file), ("column_separator", ","),
                 ("write_attribute_names", "true"),
                 ("quote_nominal_values", "true"),
                 ("format_date_attributes", "false"),
                 ("date_format", "yyyy-MM-dd HH:mm:ss"),
                 ("append_to_file", "false"), ("encoding", "UTF-8")]:
        s += param(k, v)
    s += '      </operator>\n'

    s += op_open(f"Performance ({label})", "performance_classification",
                 1251, y, VERSION)
    for k, v in [("main_criterion", "accuracy"), ("accuracy", "true"),
                 ("classification_error", "false"), ("kappa", "true"),
                 ("weighted_mean_recall", "true"),
                 ("weighted_mean_precision", "true"),
                 ("spearman_rho", "false"), ("kendall_tau", "false"),
                 ("absolute_error", "false"), ("relative_error", "false"),
                 ("relative_error_lenient", "false"),
                 ("relative_error_strict", "false"),
                 ("normalized_absolute_error", "false"),
                 ("root_mean_squared_error", "false"),
                 ("root_relative_squared_error", "false"),
                 ("squared_error", "false"), ("correlation", "false"),
                 ("squared_correlation", "false"), ("cross-entropy", "false"),
                 ("margin", "false"), ("soft_margin_loss", "false"),
                 ("logistic_loss", "false"),
                 ("skip_undefined_labels", "true"),
                 ("use_example_weights", "false")]:
        s += param(k, v)
    s += '        <list key="class_weights"/>\n'
    s += '      </operator>\n'
    return s


def connect(a: str, ap: str, b: str, bp: str) -> str:
    return (f'      <connect from_op="{escape(a)}" from_port="{ap}" '
            f'to_op="{escape(b)}" to_port="{bp}"/>\n')


def connect_result(a: str, ap: str, result_no: int) -> str:
    return (f'      <connect from_op="{escape(a)}" from_port="{ap}" '
            f'to_port="result {result_no}"/>\n')


def build(csv_path: str, pred_dir: str, models: list[str],
          mlp_balance: bool = True, neg_ratio: str = "0.1") -> str:
    x = ('<?xml version="1.0" encoding="UTF-8"?>'
         f'<process version="{VERSION}">\n')
    x += '  <context>\n    <input/>\n    <output/>\n    <macros/>\n  </context>\n'
    x += ('  <operator activated="true" class="process" '
          f'compatibility="{VERSION}" expanded="true" name="Process">\n')
    for k, v in [("logverbosity", "init"), ("random_seed", "42"),
                 ("send_mail", "never"), ("notification_email", ""),
                 ("process_duration_for_mail", "30"), ("encoding", "UTF-8")]:
        x += param(k, v)
    x += '    <process expanded="true">\n'
    x += read_csv_block(csv_path)
    x += preprocess_blocks()

    x += op_open("Multiply Training", "multiply", 780, 230, VERSION, 60)
    x += '      </operator>\n'
    x += op_open("Multiply Test", "multiply", 780, 340, VERSION, 60)
    x += '      </operator>\n'
    if "mlp" in models and mlp_balance:
        x += mlp_balance_block(MODELS["mlp"]["row"], neg_ratio)

    for i, key in enumerate(models):
        y = MODELS[key]["row"]
        x += learner_block(key, y)
        pred = f"{pred_dir}\\rm_{key}_predictions.csv"
        x += branch_blocks(key, y, pred, i + 1)

    x += connect("Read CSV", "output", "Select Attributes", "example set input")
    x += connect("Select Attributes", "example set output",
                 "Nominal to Numerical", "example set input")
    x += connect("Nominal to Numerical", "example set output",
                 LABEL_OP, "example set input")
    x += connect(LABEL_OP, "example set output",
                 "Set Role", "example set input")
    x += connect("Set Role", "example set output", "Split Data", "example set")
    x += connect("Split Data", "partition 1", "Multiply Training", "input")
    x += connect("Split Data", "partition 2", "Multiply Test", "input")

    for i, key in enumerate(models):
        label = MODELS[key]["name"]
        learner = label
        apply_ = f"Apply Model ({label})"
        write_ = f"Write CSV ({label})"
        perf_ = f"Performance ({label})"
        if key == "mlp" and mlp_balance:
            x += connect("Multiply Training", f"output {i + 1}",
                         "Sample (MLP Balance)", "example set input")
            x += connect("Sample (MLP Balance)", "example set output",
                         learner, "training set")
        else:
            x += connect("Multiply Training", f"output {i + 1}",
                         learner, "training set")
        x += connect("Multiply Test", f"output {i + 1}", apply_, "unlabelled data")
        x += connect(learner, "model", apply_, "model")
        x += connect(apply_, "labelled data", write_, "input")
        x += connect(write_, "through", perf_, "labelled data")
        x += connect_result(perf_, "performance", i + 1)

    x += '      <portSpacing port="source_input 1" spacing="0"/>\n'
    for i in range(1, len(models) + 2):
        x += f'      <portSpacing port="sink_result {i}" spacing="0"/>\n'
    x += '    </process>\n  </operator>\n</process>\n'
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV, help="path CSV (Windows) di Read CSV")
    ap.add_argument("--pred-dir", default=DEFAULT_PRED_DIR,
                    help="folder output CSV prediksi (Windows)")
    ap.add_argument("--out", default=str(RM_DIR / "proyek_4model.rmp"))
    ap.add_argument("--models", default="rf,lr,xgb,mlp")
    ap.add_argument("--tanpa-xgb", action="store_true",
                    help="buang cabang XGBoost (extension belum terpasang)")
    ap.add_argument("--mlp-cycles", type=int, default=None,
                    help="override training_cycles Neural Net (default 200)")
    ap.add_argument("--mlp-neg-ratio", default="0.1",
                    help="rasio negatif yang dipertahankan untuk MLP (default 0.1 ~ 40:1)")
    ap.add_argument("--mlp-tanpa-balance", action="store_true",
                    help="matikan Sample penyeimbang cabang MLP (data penuh)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.tanpa_xgb:
        models = [m for m in models if m != "xgb"]
    for m in models:
        if m not in MODELS:
            raise SystemExit(f"model tidak dikenal: {m}")

    if args.mlp_cycles:
        MODELS["mlp"]["params"] = [
            (k, str(args.mlp_cycles) if k == "training_cycles" else v)
            for k, v in MODELS["mlp"]["params"]]

    out = Path(args.out)
    out.write_text(build(args.csv, args.pred_dir, models,
                         mlp_balance=not args.mlp_tanpa_balance,
                         neg_ratio=args.mlp_neg_ratio), encoding="utf-8")
    print(f"[OK] {out} ({len(models)} model: {', '.join(models)})")
    for m in models:
        print(f"     prediksi -> {args.pred_dir}\\rm_{m}_predictions.csv")


if __name__ == "__main__":
    main()
