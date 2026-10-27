import numpy as np


def test_leer_3d_lee_el_npz_una_sola_vez(trabajo, tmp_path):
    # Con 11336 elevaciones de BoxingWeb, releer P3 por golpe agoto los 15 GB (23-10)
    import armar
    N, L = 6, 4
    P3 = np.arange(N * L * 17 * 3, dtype=np.float32).reshape(N, L, 17, 3)
    ruta = tmp_path / "pose3d.npz"
    np.savez_compressed(ruta, frame_dir=np.array([f"g{i // 2}" for i in range(N)]),
                        persona=np.array([i % 2 for i in range(N)]),
                        largo=np.array([4, 3, 2, 4, 1, 4]), P3=P3, M3=np.ones((N, L), bool))
    d = armar.leer_3d(ruta)
    assert d[("g1", 0)].shape == (2, 17, 3)
    np.testing.assert_array_equal(d[("g1", 0)], P3[2, :2])
    np.testing.assert_array_equal(d[("g2", 1)], P3[5])
    vals = list(d.values())
    assert vals[0].base is not None and all(v.base is vals[0].base for v in vals)
