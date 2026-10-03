"""Tests for make-package.py, the package assembler.

A release identifies each archive by its SHA512, so the same inputs must produce the same bytes,
and a model must never be packaged under the path of another file. Run with
python -m unittest discover -s scripts.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_packager():
    spec = importlib.util.spec_from_file_location("make_package", HERE / "make-package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


packager = load_packager()

F0 = "org.openvpi.otter.inference.F0"
ALIGN = "org.openvpi.otter.inference.Align"


class Assembly(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="otter-package-"))
        self.declarations = self.root / "declarations"
        self.models = self.root / "models"
        self.models.mkdir()

    def tearDown(self):
        shutil.rmtree(self.root)

    def declare(self, variant, declaration):
        package = self.declarations / variant
        directory = package / "inferences" / "x"
        directory.mkdir(parents=True)
        (package / "desc.json").write_text(json.dumps({
            "$version": "1.0", "id": f"test/{variant}", "version": "1.0.0.0",
            "compatVersion": "1.0.0.0", "runtimeLevel": 1,
            "contributions": {"inference": [{"id": "x", "path": "./inferences/x/inference.json"}]},
        }), encoding="utf-8")
        # A comment, which the JSON profile of the specification allows and a plain JSON reader
        # rejects.
        (directory / "inference.json").write_text(
            "// the declaration\n" + json.dumps(declaration), encoding="utf-8")

    def test_the_same_inputs_give_the_same_bytes(self):
        self.declare("rmvpe", {
            "interface": F0, "level": 1, "variant": "rmvpe", "name": "r",
            "exports": {"sampleRate": 16000, "interval": 0.01},
            "configuration": {"model": "../../rmvpe.onnx"},
        })
        (self.models / "rmvpe.onnx").write_bytes(bytes(range(256)) * 5000)
        archives = []
        for run in ("first", "second"):
            output = self.root / run
            output.mkdir()
            directory, _, version, _, shipped = packager.assemble(self.declarations, "rmvpe",
                                                                  self.models, output, None)
            self.assertEqual(shipped, [Path("rmvpe.onnx")])
            archives.append(packager.archive(directory, version, output).read_bytes())
        self.assertEqual(archives[0], archives[1])

    def test_the_archive_orders_its_entries_by_posix_path(self):
        # The package directory is read on whatever platform assembles the release, and sorting the
        # paths themselves would order them by the rules of that platform and fold case on Windows,
        # so one set of files would travel as two listings. The expected order is written out rather
        # than compared with a second sort, which would agree with a platform dependent key: a
        # byte order puts every upper case name before every lower case one, while folding case
        # would interleave them by letter.
        directory = self.root / "pkg"
        for name in ("a.onnx", "aa.txt", "B.onnx", "zz.txt", "Zz/d.txt"):
            member = directory / name
            member.parent.mkdir(parents=True, exist_ok=True)
            member.write_text(name, encoding="utf-8")
        output = self.root / "out"
        output.mkdir()
        with zipfile.ZipFile(packager.archive(directory, "1.0.0.0", output)) as archive:
            self.assertEqual(archive.namelist(),
                             ["pkg/B.onnx", "pkg/Zz/d.txt", "pkg/a.onnx", "pkg/aa.txt",
                              "pkg/zz.txt"])

    def test_the_archive_carries_no_trace_of_the_builder(self):
        # The creating system sits in every entry header and the library derives it from the
        # platform, so two people assembling the same release would publish two archives whose
        # digests differ. The packager pins it, and this is what the pin has to leave behind.
        self.declare("rmvpe", {
            "interface": F0, "level": 1, "variant": "rmvpe", "name": "r",
            "exports": {"sampleRate": 16000, "interval": 0.01},
            "configuration": {"model": "../../rmvpe.onnx"},
        })
        (self.models / "rmvpe.onnx").write_bytes(b"weights")
        output = self.root / "pinned"
        output.mkdir()
        directory, _, version, _, _ = packager.assemble(self.declarations, "rmvpe", self.models,
                                                        output, None)
        with zipfile.ZipFile(packager.archive(directory, version, output)) as archive:
            entries = archive.infolist()
            self.assertTrue(entries, "the archive has entries to check")
            self.assertEqual({entry.create_system for entry in entries}, {3})

    def test_the_release_version_is_the_project_version(self):
        # The manifest of a release and the packages inside it carry one version, which the project
        # states once. The packager reads it from there and the lint compares the packages with it.
        version = packager.project_version()
        self.assertRegex(version, r"^[0-9]+(\.[0-9]+){1,3}$", "the project states a version")
        root = HERE.parent / "packages"
        descriptors = sorted(root.glob("*/desc.json"))
        if not descriptors:
            self.skipTest(f"no declarations under {root}: they are no longer tracked")
        for descriptor in descriptors:
            self.assertEqual(json.loads(descriptor.read_text(encoding="utf-8"))["version"], version,
                             descriptor.parent.name)
        # The release tag names the same version without its trailing zero components, and that is
        # the name the release is published under.
        parts = version.split(".")
        while len(parts) > 1 and parts[-1] == "0":
            parts.pop()
        self.assertEqual(packager.projectversion.release_tag(version), "models-v" + ".".join(parts))

    def test_the_bundle_argument_takes_the_version_or_its_tag(self):
        # The release is published under its tag while the manifest carries the version, so the
        # argument accepts both writings of the same release and refuses anything else.
        self.assertEqual(packager.bundle_version("0.1.0.0", "0.1.0.0"), "0.1.0.0")
        self.assertEqual(packager.bundle_version("models-v0.1", "0.1.0.0"), "0.1.0.0")
        with self.assertRaises(SystemExit) as refused:
            packager.bundle_version("models-v0.2", "0.1.0.0")
        message = str(refused.exception)
        self.assertIn("0.1.0.0", message, "the version it accepts is named")
        self.assertIn("models-v0.1", message, "the tag it accepts is named")
        self.assertIn("trailing zero components", message, "the tag is explained")

    def test_a_tree_with_no_project_has_no_version_to_read(self):
        # The version of a release comes from the project, and a tree that holds no project has none
        # to read. The failure names the path it searched instead of raising on a missing file.
        with self.assertRaises(SystemExit) as refused:
            packager.project_version(self.root)
        message = str(refused.exception)
        self.assertIn(str(self.root), message)
        self.assertIn("no CMakeLists.txt declares a project version", message)

    def test_the_scripts_alone_read_the_usage_and_refuse_an_argument(self):
        # A tree that holds nothing but the scripts is how this script is used outside the
        # repository, so no path that ends before the project version is read may reach for a file
        # that tree does not have. The scripts are copied without the directory they live in.
        flat = self.root / "flat"
        flat.mkdir()
        for script in HERE.glob("*.py"):
            shutil.copy2(script, flat / script.name)

        def run(*arguments):
            return subprocess.run([sys.executable, str(flat / "make-package.py"), *arguments],
                                  capture_output=True, text=True, cwd=flat, check=False)

        result = run("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage", result.stdout)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("CMakeLists.txt", result.stdout + result.stderr)

        result = run()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("usage", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

        result = run("--variant", "rmvpe", "--models", "models", "--bundle", "models-v9.9")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("no CMakeLists.txt declares a project version at or above", result.stderr)

    def test_a_shared_file_name_is_refused(self):
        # Two dictionaries with the same name in different directories: a lookup by name would
        # package one file under the path of the other.
        self.declare("hfa", {
            "interface": ALIGN, "level": 1, "variant": "hfa", "name": "h",
            "exports": {"sampleRate": 44100, "silenceLabel": "SP"},
            "configuration": {"model": "../../model.onnx", "config": "../../config.json",
                              "vocab": "../../vocab.json", "languages": {"zxx": "zx"}},
        })
        for name in ("model.onnx", "config.json"):
            (self.models / name).write_bytes(b"{}")
        (self.models / "vocab.json").write_text(json.dumps(
            {"dictionaries": {"zx": "a/dict.txt", "yy": "b/dict.txt"}}), encoding="utf-8")
        (self.models / "dict.txt").write_text("a\ta\n", encoding="utf-8")
        output = self.root / "out"
        output.mkdir()
        with self.assertRaises(SystemExit):
            packager.assemble(self.declarations, "hfa", self.models, output, None)

        # With the package layout, each file is found at its own path.
        for directory in ("a", "b"):
            (self.models / directory).mkdir()
            (self.models / directory / "dict.txt").write_text(directory, encoding="utf-8")
        package, _, _, _, shipped = packager.assemble(self.declarations, "hfa", self.models, output,
                                                      None)
        self.assertIn(Path("a/dict.txt"), shipped)
        self.assertIn(Path("b/dict.txt"), shipped)
        self.assertEqual((package / "a" / "dict.txt").read_text(encoding="utf-8"), "a")
        self.assertEqual((package / "b" / "dict.txt").read_text(encoding="utf-8"), "b")


class Merge(unittest.TestCase):
    """The merge of one package's fragment into the manifest of a release.

    The manifest is what a host reads to decide what to install, so a merge that mixed releases
    would publish a manifest whose entries do not belong to the bundle it announces. The tests
    build the manifest they need, so they hold whatever state they check.
    """

    BUNDLE = "0.1.0.0"

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="otter-manifest-"))

    def tearDown(self):
        shutil.rmtree(self.root)

    def manifest(self, name, bundle, packages):
        path = self.root / name
        path.write_text(json.dumps({"bundleVersion": bundle, "packages": packages}),
                        encoding="utf-8")
        return path

    def fragment(self, identifier, version):
        return {
            "id": identifier,
            "file": f"{identifier.replace('/', '-')}-{version}.zip",
            "sha512": "0" * 128,
            "version": version,
            "compatVersion": version,
            "directory": identifier.replace("/", "-"),
            "size": 1,
            "models": {"model.onnx": "1" * 128},
        }

    def refusal(self, raw):
        """Writes a manifest that holds whatever is given and returns the refusal of merging into it.

        The merge is refused while it reads the manifest, so the file must be left as it was found.
        """
        path = self.root / "manifest.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(SystemExit) as refused:
            packager.merge_manifest(path, self.fragment("test/rmvpe", self.BUNDLE),
                                    "test/rmvpe", self.BUNDLE)
        self.assertEqual(path.read_text(encoding="utf-8"), json.dumps(raw),
                         "a refused merge writes nothing")
        return path, str(refused.exception)

    def test_a_manifest_without_a_packages_key_is_refused(self):
        # A merge replaces the entry of a package inside the list the manifest holds, so a manifest
        # that states no list at all cannot take one: the merge names the file and the list it
        # lacks instead of ending in the error of reading a key that is not there.
        path, message = self.refusal({"bundleVersion": self.BUNDLE})
        self.assertIn(str(path), message, "the manifest that lacks the list is named")
        self.assertIn("packages", message, "the list the manifest lacks is named")

    def test_a_manifest_whose_packages_are_null_is_refused(self):
        # A manifest that states null instead of a list is refused the same way: the merge does not
        # read it as an empty list, because a host would then install a release the manifest does
        # not announce.
        path, message = self.refusal({"bundleVersion": self.BUNDLE, "packages": None})
        self.assertIn(str(path), message, "the manifest whose list is null is named")
        self.assertIn("packages", message, "the list of the manifest is named")

    def test_a_manifest_whose_packages_are_not_a_list_is_refused(self):
        # Anything that is not a list is refused as well, so that the refusal states what the
        # manifest holds rather than what reading it into a merge ends in.
        path, message = self.refusal(
            {"bundleVersion": self.BUNDLE, "packages": "test/rmvpe"})
        self.assertIn(str(path), message, "the manifest whose packages are not a list is named")
        self.assertIn("packages", message, "the list of the manifest is named")

    def test_a_manifest_of_another_bundle_is_refused(self):
        # The manifest announces one release, and a package of another release may not be written
        # into it: a host would install a package whose version the bundle does not announce.
        path = self.manifest("manifest.json", "0.2.0.0", [])
        with self.assertRaises(SystemExit) as refused:
            packager.merge_manifest(path, self.fragment("test/rmvpe", self.BUNDLE),
                                    "test/rmvpe", self.BUNDLE)
        message = str(refused.exception)
        self.assertIn("0.2.0.0", message, "the version the manifest announces is named")
        self.assertIn(self.BUNDLE, message, "the version of this package is named")

    def test_an_entry_of_another_version_is_refused(self):
        # One manifest announces one bundle version and every entry it lists carries that version.
        # Leaving an entry of another version beside the one written here would make the manifest
        # claim two releases at once.
        path = self.manifest(
            "manifest.json", self.BUNDLE,
            [{"id": "test/other", "version": "0.2.0.0"}, {"id": "test/third", "version": "0.3.0.0"}])
        with self.assertRaises(SystemExit) as refused:
            packager.merge_manifest(path, self.fragment("test/rmvpe", self.BUNDLE),
                                    "test/rmvpe", self.BUNDLE)
        message = str(refused.exception)
        self.assertIn("test/other", message, "the id of each rejected entry is named")
        self.assertIn("test/third", message, "the id of each rejected entry is named")
        self.assertIn("0.2.0.0", message, "the version of the rejected entry is named")
        self.assertIn(self.BUNDLE, message, "the version of this package is named")

    def test_a_manifest_of_the_same_bundle_takes_the_entry(self):
        # The normal path: the entry of the package is replaced in place of being added, and the
        # entries of the manifest stay ordered by id so that two assemblies of one release agree.
        path = self.manifest("manifest.json", self.BUNDLE,
                             [{"id": "test/other", "version": self.BUNDLE},
                              {"id": "test/rmvpe", "version": self.BUNDLE}])
        packager.merge_manifest(path, self.fragment("test/rmvpe", self.BUNDLE), "test/rmvpe",
                               self.BUNDLE)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["bundleVersion"], self.BUNDLE)
        self.assertEqual([entry["id"] for entry in manifest["packages"]],
                         ["test/other", "test/rmvpe"])
        self.assertEqual(manifest["packages"][1], self.fragment("test/rmvpe", self.BUNDLE),
                         "the current package is what the entry carries")

    def test_a_missing_manifest_is_created_with_the_bundle(self):
        # The first package of a release creates the manifest, which announces the version every
        # package of that release is built from.
        path = self.root / "manifest.json"
        fragment = self.fragment("test/rmvpe", self.BUNDLE)
        packager.merge_manifest(path, fragment, "test/rmvpe", self.BUNDLE)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["bundleVersion"], self.BUNDLE)
        self.assertEqual([entry["id"] for entry in manifest["packages"]], ["test/rmvpe"])
        self.assertEqual(manifest["packages"][0], fragment)
        self.assertEqual(path.read_text(encoding="utf-8"),
                         json.dumps(manifest, indent=4) + "\n",
                         "the manifest is written the way the packager writes it")


if __name__ == "__main__":
    unittest.main()
