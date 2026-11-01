#!/usr/bin/env python3
"""Offline parser/verification tests. These are not an independent reference-header ABI result."""
import importlib.util
import hashlib
from html import escape
import json
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
class SnapshotIntegrationTests(unittest.TestCase):
    pages={'NvInferRuntime.h':'_nv_infer_runtime_8h_source.html','NvInferImpl.h':'_nv_infer_impl_8h_source.html','NvInferRuntimeBase.h':'_nv_infer_runtime_base_8h_source.html','NvInferRuntimeCommon.h':'_nv_infer_runtime_common_8h_source.html','NvInferPluginBase.h':'_nv_infer_plugin_base_8h_source.html','NvInferLegacyDims.h':'_nv_infer_legacy_dims_8h_source.html','NvInferVersion.h':'_nv_infer_version_8h_source.html'}
    def fixture(self,root):
        html=root/'html';html.mkdir();output=root/'snapshot'
        for name,page in self.pages.items():
            source='// SPDX-License-Identifier: Apache-2.0\n#pragma once\n'
            if name=='NvInferRuntime.h':source+='#include "NvInferTestDependency.h"\n#include <cuda_runtime_api.h>\n'
            if name=='NvInferVersion.h':source+='#define NV_TENSORRT_MAJOR 1\n#define NV_TENSORRT_MINOR 7\n#define NV_TENSORRT_PATCH 1\n'
            self.page(html/page,source)
        self.page(html/'_nv_infer_test_dependency_8h_source.html','// SPDX-License-Identifier: Apache-2.0\n#include "NvInferRuntime.h"\n')
        return html,output
    def page(self,path,source,version='1.7.1'):
        lines=''.join(f'<div class="line"><span class="lineno">{i}</span>{escape(line)}</div>' for i,line in enumerate(source.splitlines(),1))
        path.write_text(f'<div id="projectname">TensorRT for RTX<span id="projectnumber">&#160;{version}</span></div>'+lines)
    def run_snapshot(self,html,output):
        return subprocess.run([sys.executable,str(crate/'tools/snapshot_public_headers.py'),'--html-dir',str(html),'--output-dir',str(output)],capture_output=True,text=True)
    def test_complete_offline_snapshot_dependencies_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            html,output=self.fixture(Path(directory));result=self.run_snapshot(html,output)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(len(manifest['headers']),8)
            self.assertEqual(manifest['external_includes'],['cuda_runtime_api.h'])
            for name,record in manifest['headers'].items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(),record['sha256'])
            # Hash mismatch must reject before attempting to run a compiler.
            (output/'NvInferRuntime.h').write_text('changed')
            result=subprocess.run([sys.executable,str(crate/'tools/check_reference_abi.py'),'--reference-dir',str(output),'--manifest',str(output/'manifest.json'),'--native-library','missing','--abi-library','missing','--output-dir',str(output/'client'),'--cxx','must-not-run'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('hash mismatch',result.stderr)
    def test_mixed_version_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            html,output=self.fixture(Path(directory));path=html/self.pages['NvInferImpl.h']
            path.write_text(path.read_text().replace('1.7.1','1.6.1'))
            result=self.run_snapshot(html,output);self.assertNotEqual(result.returncode,0)
            self.assertIn('consistently labeled',result.stderr);self.assertFalse((output/'manifest.json').exists())
    def test_body_version_mention_does_not_override_project_version(self):
        with tempfile.TemporaryDirectory() as directory:
            html,output=self.fixture(Path(directory));path=html/self.pages['NvInferImpl.h']
            path.write_text(path.read_text().replace('1.7.1','1.6.1')+'<p>TensorRT for RTX 1.7.1</p>')
            result=self.run_snapshot(html,output);self.assertNotEqual(result.returncode,0);self.assertFalse(output.exists())
    def test_wrong_version_macros_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            html,output=self.fixture(Path(directory));path=html/self.pages['NvInferVersion.h']
            path.write_text(path.read_text().replace('NV_TENSORRT_PATCH 1','NV_TENSORRT_PATCH 2'))
            result=self.run_snapshot(html,output);self.assertNotEqual(result.returncode,0)
            self.assertIn('version macros',result.stderr);self.assertFalse(output.exists())
    def test_existing_snapshot_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            html,output=self.fixture(Path(directory));output.mkdir();sentinel=output/'manifest.json';sentinel.write_text('existing snapshot')
            result=self.run_snapshot(html,output);self.assertNotEqual(result.returncode,0)
            self.assertIn('fresh snapshot path',result.stderr);self.assertEqual(sentinel.read_text(),'existing snapshot')
    def test_bad_license_publishes_no_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            html,output=self.fixture(Path(directory));path=html/self.pages['NvInferImpl.h']
            path.write_text(path.read_text().replace('Apache-2.0','unreviewed'))
            result=self.run_snapshot(html,output);self.assertNotEqual(result.returncode,0);self.assertFalse(output.exists())
    def test_missing_dependency_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            html,output=self.fixture(Path(directory));(html/'_nv_infer_test_dependency_8h_source.html').unlink()
            result=self.run_snapshot(html,output);self.assertNotEqual(result.returncode,0)
            self.assertFalse((output/'manifest.json').exists())
if __name__=='__main__':unittest.main()
