import numpy as np

from experiments.prepare_mh1m import prepare_mh1m


METADATA_COLUMNS = np.array(
    [
        "VT_LAST_ANALYSIS_DATE",
        "VT_SIZE",
        "SHA256",
        "VT_MD5",
        "VT_TIMES_SUBMITTED",
        "VT_SCANNERS_FAILURE",
        "VT_SCANNERS_MALICIOUS",
        "VT_SCANNERS_UNDETECTED",
        "VT_SCANNERS_SUGGESTED_THREAT_LABEL",
        "VT_SCANNERS_NAMES",
        "CLASS",
    ],
    dtype=object,
)


def test_prepare_mh1m_groups_rows_and_appends_binary_label(tmp_path):
    source_path = tmp_path / "mh1m.npz"
    output_dir = tmp_path / "monthly"
    features = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int8
    )
    metadata = np.empty((4, len(METADATA_COLUMNS)), dtype=object)
    metadata[:] = None
    metadata[:, 0] = [
        "2023-02-01T12:00:00Z",
        "2023-01-31T23:59:59Z",
        "2023-02-28T01:02:03Z",
        "2023-03-01T00:00:00Z",
    ]
    metadata[:, 2] = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
    metadata[:, 10] = [0, 1, 1, 0]
    np.savez_compressed(
        source_path,
        data=features,
        column_names=np.array(["feature_a", "feature_b"], dtype=object),
        metadata=metadata,
        metadata_columns=METADATA_COLUMNS,
        sha256=metadata[:, 2],
    )

    manifest = prepare_mh1m(source_path, output_dir, chunk_size=1)

    assert manifest["samples"] == 4
    assert manifest["observed_month_count"] == 3
    assert manifest["missing_months"] == []
    assert manifest["splits"]["2023-02"]["malware"] == 1

    with np.load(output_dir / "2023-02_selected.npz", allow_pickle=False) as split:
        assert split.files == [
            "data",
            "column_names",
            "sha256",
            "last_analysis_time",
        ]
        np.testing.assert_array_equal(
            split["data"], np.array([[1, 2, 0], [5, 6, 1]], dtype=np.int8)
        )
        np.testing.assert_array_equal(
            split["column_names"], ["feature_a", "feature_b", "CLASS"]
        )
        assert split["data"].dtype == np.int8
        assert split["sha256"].tolist() == ["a" * 64, "c" * 64]
        assert split["last_analysis_time"].dtype == np.dtype("datetime64[s]")
