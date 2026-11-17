"""
ordinal.py — Regresi ordinal via dekomposisi biner kumulatif
=============================================================
Regresi ordinal memodelkan target BERURUTAN (0 < 1 < 2) tanpa menganggap
jarak antar kelas setara dan tanpa membuang informasi urutan seperti
klasifikasi multikelas biasa.

Implementasi: cumulative binary decomposition (K-1 model biner).
  P(Y > 0) = P(Sedang atau Tinggi)   -> model biner #1
  P(Y > 1) = P(Tinggi)               -> model biner #2
  P(Y = 0) = 1 - P(Y > 0)
  P(Y = 1) = P(Y > 0) - P(Y > 1)
  P(Y = 2) = P(Y > 1)

Keunggulan dibanding multikelas:
  - model hanya perlu memutuskan "sudah lewat ambang k atau belum",
    sehingga informasi urutan dipakai langsung;
  - setiap model biner dilatih pada seluruh data (bukan hanya subset kelas);
  - jumlah parameter lebih sedikit -> lebih stabil pada data kecil (n=610).

Monotonisitas: secara teori P(Y>0) >= P(Y>1). Dua model dilatih terpisah
sehingga hasilnya bisa melanggar; probabilitas kumulatif dipaksa turun
monoton lewat cumulative minimum sebelum diubah menjadi probabilitas kelas.

Catatan istilah: ini regresi ordinal, bukan klasifikasi multikelas.
Metrik utamanya QWK (quadratic weighted kappa) yang memberi penalti
kuadratik pada kesalahan jarak kelas.
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class OrdinalCumulative:
    """Wrapper regresi ordinal untuk estimator biner sklearn-compatible."""

    def __init__(self, estimator, n_classes=3):
        self.estimator = estimator
        self.n_classes = n_classes

    def fit(self, X, y):
        y = np.asarray(y)
        self.estimators_ = []
        for k in range(self.n_classes - 1):
            est = clone(self.estimator)
            est.fit(X, (y > k).astype(int))
            self.estimators_.append(est)
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict_cumulative(self, X):
        """P(Y > k) untuk k = 0..K-2, dipaksa monoton turun."""
        cum = np.column_stack([e.predict_proba(X)[:, 1] for e in self.estimators_])
        return np.minimum.accumulate(cum, axis=1)

    def predict_proba(self, X):
        cum = self.predict_cumulative(X)
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        proba = np.empty((n, self.n_classes))
        proba[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, self.n_classes - 1):
            proba[:, k] = cum[:, k - 1] - cum[:, k]
        proba[:, -1] = cum[:, -1]
        proba = np.clip(proba, 0.0, 1.0)
        # normalisasi kecil untuk sisa pembulatan
        s = proba.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return proba / s

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def ordinal_metrics(y_true, y_pred, n_classes=3):
    """Metrik evaluasi regresi ordinal."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    diff = np.abs(y_true - y_pred)
    return {
        "qwk": round(float(cohen_kappa_score(
            y_true, y_pred, weights="quadratic", labels=labels)), 4),
        "mae_class": round(float(diff.mean()), 4),
        "exact_acc": round(float((diff == 0).mean()), 4),
        "adjacent_acc": round(float((diff <= 1).mean()), 4),
        "per_class": {
            f"class_{c}": {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(sup[i]),
            }
            for i, c in enumerate(labels)
        },
        "confusion": cm.tolist(),
        "n": int(len(y_true)),
    }


def predict_warning(proba, t_high=0.25, t_mid=0.40):
    """Aturan keputusan sadar-biaya untuk peringatan dini.

    argmax memilih kelas dengan probabilitas tertinggi, padahal dalam
    peringatan tsunami melewatkan kelas Awas jauh lebih mahal daripada
    satu alarm palsu. Aturan ini menurunkan ambang untuk kelas atas:

        Awas    jika P(Awas) >= t_high
        Siaga   jika P(Siaga) + P(Awas) >= t_mid
        Waspada selainnya

    Nilai default dipilih dari out-of-fold CV (lihat .report/modeling.md).
    """
    proba = np.asarray(proba)
    p_high = proba[:, 2]
    p_mid_up = proba[:, 1] + proba[:, 2]
    return np.where(p_high >= t_high, 2, np.where(p_mid_up >= t_mid, 1, 0))
