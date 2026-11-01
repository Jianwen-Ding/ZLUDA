#!/usr/bin/env python3
"""Offline parser/verification tests. These are not an independent reference-header ABI result."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

crate=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('snapshot',crate/'tools/snapshot_public_headers.py');snapshot=importlib.util.module_from_spec(spec);spec.loader.exec_module(snapshot)
class ParserTests(unittest.TestCase):
    def test_doxygen_lines_and_escaping(self):
        html='<div class="line"><span class="lineno"><a>1</a></span>// SPDX-License-Identifier: Apache-2.0</div><div class="line"><span class="lineno">2</span>&#160;virtual <a>Type</a>* f() &amp; = 0;</div>'
        self.assertEqual(snapshot.extract(html),'// SPDX-License-Identifier: Apache-2.0\n virtual Type* f() & = 0;\n')
    def test_reject_missing_source(self):
        with self.assertRaises(ValueError):snapshot.extract('<p>SPDX-License-Identifier: Apache-2.0</p>')
    def test_reject_unreviewed_license(self):
        with self.assertRaises(ValueError):snapshot.extract('<div class="line">// unknown license</div>')
    def test_missing_snapshot_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'manifest.json';manifest.write_text('{"target":"TensorRT-RTX 1.7.1","headers":{}}')
            result=subprocess.run([sys.executable,str(crate/'tools/check_reference_abi.py'),'--reference-dir',directory,'--manifest',str(manifest),'--native-library','missing','--abi-library','missing','--output-dir',directory],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('missing required reference headers',result.stderr)
if __name__=='__main__':unittest.main()
