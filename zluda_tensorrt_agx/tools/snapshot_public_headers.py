#!/usr/bin/env python3
"""Extract Apache-2.0 RTX interface source from public Doxygen pages, with hashes and a fixed version check.
No SDK, shared library, engine or proprietary binary is downloaded. Network access is needed unless HTML
pages are supplied explicitly with --html-dir. Missing/mixed-version pages fail rather than falling back.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.request
import tempfile

HEADERS=['NvInferRuntime.h','NvInferImpl.h','NvInferRuntimeBase.h','NvInferRuntimeCommon.h','NvInferPluginBase.h','NvInferLegacyDims.h','NvInferVersion.h']
class SourceParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.lines=[];self.depth=0;self.skip=0;self.parts=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='div':
            if self.depth:self.depth+=1
            elif 'line' in attrs.get('class','').split():self.depth=1;self.parts=[]
        if self.depth and tag=='span':
            if self.skip or 'lineno' in attrs.get('class','').split():self.skip+=1
    def handle_endtag(self,tag):
        if self.depth and tag=='span' and self.skip:self.skip-=1
        if self.depth and tag=='div':
            self.depth-=1
            if not self.depth:self.lines.append(''.join(self.parts).replace('\xa0',' ').rstrip())
    def handle_data(self,text):
        if self.depth and not self.skip:self.parts.append(text)
class VersionParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.depth=0;self.parts=[];self.labels=[]
    def handle_starttag(self,tag,attrs):
        if tag=='div':
            if self.depth:self.depth+=1
            elif dict(attrs).get('id')=='projectname':self.depth=1;self.parts=[]
    def handle_endtag(self,tag):
        if tag=='div' and self.depth:
            self.depth-=1
            if not self.depth:self.labels.append(''.join(self.parts))
    def handle_data(self,text):
        if self.depth:self.parts.append(text)
def target_version(html):
    parser=VersionParser();parser.feed(html)
    return len(parser.labels)==1 and bool(re.fullmatch(r'\s*TensorRT\s+for\s+RTX\s+1\.7\.1\s*',parser.labels[0]))

def extract(html):
    parser=SourceParser();parser.feed(html);source='\n'.join(parser.lines)+'\n'
    if not parser.lines or 'SPDX-License-Identifier: Apache-2.0' not in source:raise ValueError('missing public Apache-2.0 source lines')
    return source

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--html-dir',type=Path,help='Previously obtained public documentation HTML pages only')
    a=p.parse_args()
    if a.output_dir.exists():p.error('output directory must not exist; choose a fresh snapshot path')
    records={};sources={};pending=list(HEADERS);external=set()
    while pending:
        name=pending.pop(0)
        if name in records:continue
        page='_'+re.sub(r'(?<!^)(?=[A-Z])','_',Path(name).stem).lower()+'_8h_source.html'
        url='https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/'+page
        html=(a.html_dir/page).read_text() if a.html_dir else urllib.request.urlopen(url,timeout=30).read().decode('utf-8')
        if not target_version(html):raise ValueError(f'{name}: not a consistently labeled RTX 1.7.1 page')
        source=extract(html);data=source.encode();sources[name]=data
        records[name]={'url':url,'sha256':hashlib.sha256(data).hexdigest(),'html_sha256':hashlib.sha256(html.encode()).hexdigest()}
        for include in re.findall(r'^\s*#\s*include\s*[<"]([^>"\n]+)[>"]',source,re.M):
            if re.fullmatch(r'NvInfer\w*\.h',include):pending.append(include)
            else:external.add(include)
    version=sources['NvInferVersion.h'].decode()
    for key,value in [('MAJOR',1),('MINOR',7),('PATCH',1)]:
        if not re.search(r'#\s*define\s+NV_TENSORRT_'+key+r'\s+'+str(value)+r'\b',version):raise ValueError('version macros do not match RTX 1.7.1')
    manifest={'target':'TensorRT-RTX 1.7.1','headers':records,'external_includes':sorted(external),'origin':'Apache-2.0 public documentation source; extracted source is otherwise unmodified'}
    a.output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.public-headers-',dir=a.output_dir.parent) as temporary:
        staging=Path(temporary)
        for name,data in sources.items():(staging/name).write_bytes(data)
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        staging.rename(a.output_dir)
    print(f'Extracted {len(records)} public headers; compile/run check remains required')
if __name__=='__main__':main()
