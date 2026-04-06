import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hapke_mcmc_package.etl.ingestion import DataManager


class DummyResponse:
    def __init__(self, payload: bytes = b"test", text: str = ""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        yield self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.manifest = self.root / "survey_manifest.csv"
        self.data_root = self.root / "data"
        (self.data_root / "01_calibrated_images").mkdir(parents=True, exist_ok=True)
        (self.data_root / "02_spice_kernels").mkdir(parents=True, exist_ok=True)
        (self.data_root / "03_dtm").mkdir(parents=True, exist_ok=True)

        with self.manifest.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image_filename"])
            writer.writerow(["A.IMG"])
            writer.writerow(["B.IMG"])

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_validate_data_ready_false_when_images_or_tm_missing(self):
        manager = DataManager(str(self.manifest), str(self.data_root))
        self.assertFalse(manager.validate_data_ready())

    def test_validate_data_ready_true_when_files_exist(self):
        (self.data_root / "01_calibrated_images" / "A.IMG").touch()
        (self.data_root / "01_calibrated_images" / "B.IMG").touch()
        (self.data_root / "02_spice_kernels" / "vesta.tm").touch()

        manager = DataManager(str(self.manifest), str(self.data_root))
        self.assertTrue(manager.validate_data_ready())

    @patch("hapke_mcmc_package.etl.ingestion.requests.get")
    def test_download_missing_data_downloads_img_and_lbl(self, mock_get):
        (self.data_root / "01_calibrated_images" / "A.IMG").touch()
        mock_get.return_value = DummyResponse(payload=b"ok")

        manager = DataManager(
            str(self.manifest),
            str(self.data_root),
            pds_base_url="https://example.test/",
        )
        with patch.object(Path, "resolve", return_value=Path("/scratch/test_data")):
            manager.download_missing_data()

        self.assertTrue((self.data_root / "01_calibrated_images" / "B.IMG").exists())
        self.assertTrue((self.data_root / "01_calibrated_images" / "B.LBL").exists())
        self.assertGreaterEqual(mock_get.call_count, 2)

    def test_construct_image_url_builds_candidate_paths(self):
        manager = DataManager(str(self.manifest), str(self.data_root))
        img_urls, lbl_urls = manager._construct_image_url("FC21A0001234")

        self.assertTrue(any(url.endswith("FC21A0001234.IMG") for url in img_urls))
        self.assertTrue(any(url.endswith("FC21A0001234.LBL") for url in lbl_urls))
        self.assertTrue(img_urls[0].startswith("https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVFC2_1B/DATA/IMG/"))

    def test_extract_kernel_paths_from_metakernel(self):
        manager = DataManager(str(self.manifest), str(self.data_root))
        metakernel_text = """
KERNELS_TO_LOAD = (
   '$KERNELS/spk/de440s.bsp',
   '$KERNELS/ck/dawn_fc.bc',
   '$KERNELS/fk/dawn_vesta.tf',
   '$KERNELS/ik/fc2.ti',
   '$KERNELS/sclk/dawn.tsc'
)
"""
        refs = manager._extract_kernel_paths_from_metakernel(metakernel_text)
        self.assertIn("spk/de440s.bsp", refs)
        self.assertIn("ck/dawn_fc.bc", refs)
        self.assertIn("fk/dawn_vesta.tf", refs)
        self.assertIn("ik/fc2.ti", refs)
        self.assertIn("sclk/dawn.tsc", refs)

    def test_construct_kernel_url_maps_extensions(self):
        manager = DataManager(str(self.manifest), str(self.data_root))
        self.assertTrue(manager._construct_kernel_url("de440s.bsp").endswith("/spk/de440s.bsp"))
        self.assertTrue(manager._construct_kernel_url("dawn_fc.bc").endswith("/ck/dawn_fc.bc"))
        self.assertTrue(manager._construct_kernel_url("dawn_vesta.tf").endswith("/fk/dawn_vesta.tf"))
        self.assertTrue(manager._construct_kernel_url("fc2.ti").endswith("/ik/fc2.ti"))
        self.assertTrue(manager._construct_kernel_url("dawn.tsc").endswith("/sclk/dawn.tsc"))

    @patch("hapke_mcmc_package.etl.ingestion.Path.resolve", return_value=Path("/scratch/test_data"))
    @patch("hapke_mcmc_package.etl.ingestion.DataManager._download_file_with_retries")
    @patch("hapke_mcmc_package.etl.ingestion.DataManager._discover_metakernel_urls")
    def test_download_spice_kernels_downloads_metakernel_and_children(
        self,
        mock_discover,
        mock_download,
        _mock_resolve,
    ):
        manager = DataManager(str(self.manifest), str(self.data_root))
        mock_discover.return_value = ["https://naif.jpl.nasa.gov/pub/naif/DAWN/kernels/mk/vesta_survey.tm"]

        def fake_download(url, destination, max_retries=3):
            if destination.suffix.lower() == ".tm":
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    "KERNELS_TO_LOAD=( '$KERNELS/spk/a.bsp', '$KERNELS/ck/b.bc', '$KERNELS/fk/c.tf' )",
                    encoding="utf-8",
                )
                return True
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"ok")
            return True

        mock_download.side_effect = fake_download
        ok = manager.download_spice_kernels()
        self.assertTrue(ok)
        self.assertTrue((self.data_root / "02_spice_kernels" / "vesta_survey.tm").exists())
        self.assertTrue((self.data_root / "02_spice_kernels" / "a.bsp").exists())
        self.assertTrue((self.data_root / "02_spice_kernels" / "b.bc").exists())
        self.assertTrue((self.data_root / "02_spice_kernels" / "c.tf").exists())

    @patch("hapke_mcmc_package.etl.ingestion.time.sleep", return_value=None)
    @patch("hapke_mcmc_package.etl.ingestion.requests.get")
    def test_download_retries_until_success(self, mock_get, _mock_sleep):
        class RequestError(Exception):
            pass

        from requests.exceptions import RequestException

        mock_get.side_effect = [RequestException("drop"), RequestException("drop"), DummyResponse()]
        manager = DataManager(str(self.manifest), str(self.data_root))

        target = self.data_root / "01_calibrated_images" / "retry.IMG"
        manager._download_file_with_retries("https://example.test/retry.IMG", target, max_retries=3)

        self.assertTrue(target.exists())
        self.assertEqual(mock_get.call_count, 3)

    @patch("hapke_mcmc_package.etl.ingestion.Path.resolve", return_value=Path("/scratch/test_data"))
    @patch("hapke_mcmc_package.etl.ingestion.DataManager._list_remote_directory_filtered")
    @patch("hapke_mcmc_package.etl.ingestion.DataManager._download_file_with_retries")
    def test_download_dtm_geometry_metadata_uses_exact_archive_path(
        self,
        mock_download,
        mock_list,
        _mock_resolve,
    ):
        manager = DataManager(str(self.manifest), str(self.data_root))

        img_url = "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVSPG_2/DATA/SHAPE/VESTA_HAMO_DTM_93M.IMG"
        mock_list.return_value = ([img_url], [])

        def fake_download(url, destination, max_retries=3):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"ok")
            return True

        mock_download.side_effect = fake_download
        ok = manager._find_and_download_dtm(max_depth=1, max_dirs=1)

        self.assertTrue(ok)
        self.assertTrue((self.data_root / "03_dtm" / "VESTA_HAMO_DTM_93M.IMG").exists())
        self.assertTrue((self.data_root / "03_dtm" / "dawn_vesta_SPG20160901.lbl").exists())
        self.assertTrue((self.data_root / "03_dtm" / "dawn_vesta_SPG20160901.tpc").exists())
        called_urls = [call.args[0] for call in mock_download.call_args_list]
        self.assertIn("https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVSPG_2/GEOMETRY/dawn_vesta_SPG20160901.lbl", called_urls)
        self.assertIn("https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVSPG_2/GEOMETRY/dawn_vesta_SPG20160901.tpc", called_urls)

    @patch("hapke_mcmc_package.etl.ingestion.Path.resolve", return_value=Path("/scratch/test_data"))
    @patch("hapke_mcmc_package.etl.ingestion.DataManager._list_remote_directory_filtered")
    @patch("hapke_mcmc_package.etl.ingestion.DataManager._download_file_with_retries")
    def test_find_and_download_dtm_removes_orphaned_img_before_crawl(
        self,
        mock_download,
        mock_list,
        _mock_resolve,
    ):
        manager = DataManager(str(self.manifest), str(self.data_root))

        orphan = self.data_root / "03_dtm" / "ORPHAN_DTM.IMG"
        orphan.write_bytes(b"old")

        img_url = "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVSPG_2/DATA/SHAPE/VESTA_HAMO_DTM_93M.IMG"
        mock_list.return_value = ([img_url], [])

        def fake_download(url, destination, max_retries=3):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"ok")
            if destination.suffix.upper() == ".LBL":
                destination.write_text("label", encoding="utf-8")
            return True

        mock_download.side_effect = fake_download
        ok = manager._find_and_download_dtm(max_depth=1, max_dirs=1)

        self.assertTrue(ok)
        self.assertFalse(orphan.exists())

    @patch("hapke_mcmc_package.etl.ingestion.Path.resolve", return_value=Path("/scratch/test_data"))
    @patch("hapke_mcmc_package.etl.ingestion.DataManager._download_file_with_retries")
    def test_find_and_download_dtm_preserves_existing_imgs_when_geometry_ready(
        self,
        mock_download,
        _mock_resolve,
    ):
        manager = DataManager(str(self.manifest), str(self.data_root))

        img = self.data_root / "03_dtm" / "VE_HAMO_G_00N_330E_EQU_DTM.IMG"
        img.write_bytes(b"existing")
        (self.data_root / "03_dtm" / "dawn_vesta_SPG20160901.lbl").write_text("lbl", encoding="utf-8")
        (self.data_root / "03_dtm" / "dawn_vesta_SPG20160901.tpc").write_text("tpc", encoding="utf-8")

        ok = manager._find_and_download_dtm(max_depth=1, max_dirs=1)

        self.assertTrue(ok)
        self.assertTrue(img.exists())
        mock_download.assert_not_called()

    @patch("hapke_mcmc_package.etl.ingestion.Path.resolve", return_value=Path("/scratch/test_data"))
    @patch("hapke_mcmc_package.etl.ingestion.DataManager._list_remote_directory_filtered")
    @patch("hapke_mcmc_package.etl.ingestion.DataManager._download_file_with_retries")
    def test_find_and_download_dtm_raises_when_geometry_metadata_unavailable(
        self,
        mock_download,
        mock_list,
        _mock_resolve,
    ):
        manager = DataManager(str(self.manifest), str(self.data_root))

        img_url = "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVSPG_2/DATA/SHAPE/VESTA_HAMO_DTM_93M.IMG"
        mock_list.return_value = ([img_url], [])

        def fake_download(url, destination, max_retries=3):
            if url.endswith(".IMG"):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"ok")
                return True
            return False

        mock_download.side_effect = fake_download
        with self.assertRaises(RuntimeError):
            manager._find_and_download_dtm(max_depth=1, max_dirs=1)

        self.assertTrue((self.data_root / "03_dtm" / "VESTA_HAMO_DTM_93M.IMG").exists())
        self.assertFalse((self.data_root / "03_dtm" / "dawn_vesta_SPG20160901.lbl").exists())
        self.assertFalse((self.data_root / "03_dtm" / "dawn_vesta_SPG20160901.tpc").exists())


if __name__ == "__main__":
    unittest.main()
