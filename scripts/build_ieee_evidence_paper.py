"""Build a 20-page research manuscript from preserved experiment artifacts.

No training, inference, backend change, or checkpoint mutation is performed.
Run from the repository root with .venv/Scripts/python.exe.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from copy import deepcopy
from html import escape
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, Image, Flowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION_START
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/word_accuracy/avhubert_miracl8_v1'
OUT=ROOT/'submission_package/ieee_research_paper'
ASSETS=OUT/'figures'
SOURCE=ROOT/'submission_package/source/IEEE_Research_Paper_20_Pages.md'
TITLE='Context-Aware Lip Reading and Speech Generation Using Deep Learning'
NAME='LipNet_IEEE_Research_Paper_20_Pages'
AUTHORS='Ravikumar Gupta, Omkar Gupta, and Prof. Umang Patel'
AFFILIATIONS=['Department of Electronics and Computer Engineering',
              'K. J. Somaiya School of Engineering, Somaiya Vidyavihar University',
              'Mumbai, India',
              'Email: {ravikumar.g, omkar.gupta, umang.patel}@somaiya.edu']
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-half,mid+half]

def evidence():
    e=load(RUN/'evaluation.json');s=load(RUN/'selection.json');p=load(RUN/'paired_comparison.json');protocol=load(RUN/'protocol.json')
    old=rows(RUN/'preserved_scratch_test_predictions.jsonl');new=rows(RUN/'test_predictions.jsonl')
    oldby={r['id']:r for r in old};newby={r['id']:r for r in new};v=e['vocabulary']
    assert len(oldby)==len(newby)==192 and oldby.keys()==newby.keys()
    assert not set(newby)&set(protocol['quarantined_previous_test_ids'])
    assert sum(r['correct_raw'] for r in old)==24 and sum(r['correct_raw'] for r in new)==182
    matrices=[]
    for rr in [old,new]:
        m=np.zeros((8,8),int)
        for r in rr:
            assert r['ground_truth']==oldby[r['id']]['ground_truth']
            m[v.index(r['ground_truth']),v.index(r['candidate'])]+=1
        matrices.append(m)
    assert matrices[1].tolist()==e['confusion_matrix']
    assert sha(RUN/'head.pt')==e['head_sha256']==p['pretrained_head_sha256']
    assert sha(ROOT/'runs/multidataset/miracl8_cpu_v2/best.pt')==p['scratch_checkpoint_sha256']
    assert sha(ROOT/p['scratch_model_definition'])==p['scratch_model_source_sha256']
    assert sha(RUN/'features.jsonl')==s['features_manifest_sha256']
    assert len(s['candidates'])==27 and s['test_used'] is False
    assert sum(v.index(r['ground_truth']) in np.argsort(r['probabilities'])[-5:] for r in new)==191
    b=sum(not oldby[k]['correct_raw'] and newby[k]['correct_raw'] for k in oldby)
    c=sum(oldby[k]['correct_raw'] and not newby[k]['correct_raw'] for k in oldby)
    assert (b,c)==(158,0)
    summary={'paired_ids':192,'before_correct':24,'after_correct':182,'difference_pp':100*(182-24)/192,
             'wilson_95_before':wilson(24,192),'wilson_95_after':wilson(182,192),
             'paired_discordant':[b,c],'mcnemar_exact_descriptive_p':2*0.5**158,
             'statistical_caveat':'Clip-level descriptive calculations; speaker dependence and pilot exposure preclude a population-independent significance claim.',
             'current_policy':load(RUN/'rejection_policy.json'),'evaluated_policy':e['rejection_policy'],
             'source_hashes':{str(x.relative_to(ROOT)):sha(x) for x in [RUN/n for n in ['evaluation.json','selection.json','paired_comparison.json','protocol.json','test_predictions.jsonl','preserved_scratch_test_predictions.jsonl']]}}
    (OUT/'evidence_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return e,s,p,protocol,old,new,matrices,summary

def figures(e,s,p,protocol,old,new,matrices,summary):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    names=[]
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(ASSETS/f'{name}.{ext}',dpi=260,bbox_inches='tight',facecolor='white')
        plt.close(fig);names.append(name)
    v=e['vocabulary'];blue='#245778';teal='#248176';gray='#818791'
    fig,ax=plt.subplots(figsize=(3.5,2.5),layout='constrained')
    vals=np.array([24,182])/192*100;cis=np.array([wilson(24,192),wilson(182,192)])*100
    ax.bar(['Scratch\nCNN–BiGRU','Frozen encoder\n+ linear head'],vals,color=[gray,teal],width=.55)
    ax.errorbar([0,1],vals,yerr=[vals-cis[:,0],cis[:,1]-vals],fmt='none',color='black',capsize=4)
    for i,x in enumerate(vals):ax.text(i,x+4,f'{x:.2f}%',ha='center',fontsize=9)
    ax.set_ylim(0,110);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Top-1 accuracy (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'paired_accuracy')
    for label,m in zip(['baseline','transfer'],matrices):
        fig,ax=plt.subplots(figsize=(3.5,3.4),layout='constrained');ax.imshow(m,cmap='Blues',vmin=0,vmax=24)
        ax.set_xticks(range(8),v,rotation=50,ha='right',fontsize=7);ax.set_yticks(range(8),v,fontsize=7)
        ax.set_xlabel('Predicted class');ax.set_ylabel('Reference class')
        for i in range(8):
            for j in range(8):ax.text(j,i,str(m[i,j]),ha='center',va='center',fontsize=7,color='white' if m[i,j]>12 else 'black')
        save(fig,'confusion_'+label)
    fig,ax=plt.subplots(figsize=(3.5,2.7),layout='constrained');y=np.arange(8)
    ax.barh(y-.16,np.diag(matrices[0])/24*100,height=.3,label='Scratch',color=gray)
    ax.barh(y+.16,np.diag(matrices[1])/24*100,height=.3,label='Transfer',color=teal)
    ax.set_yticks(y,v,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xlabel('Class recall (%)');ax.legend(fontsize=7,loc='lower right');save(fig,'per_word')
    fig,ax=plt.subplots(figsize=(3.5,2.3),layout='constrained');speakers=list(e['speakers'])
    vals=[100*e['speakers'][x]['accuracy'] for x in speakers];ax.bar(speakers,vals,color=blue,width=.55)
    for i,x in enumerate(speakers):ax.text(i,vals[i]+2,f"{e['speakers'][x]['correct']}/64",ha='center')
    ax.set_ylim(0,110);ax.set_ylabel('Test accuracy (%)');save(fig,'speakers')
    grid=np.array([[next(c['validation_accuracy']*100 for c in s['candidates'] if c['layer']==l and c['pooling']==pool and c['C']==C) for C in [.01,.1,1]] for l in [6,9,12] for pool in ['mean','temporal3','mean_std']])
    fig,ax=plt.subplots(figsize=(3.5,3.5),layout='constrained');ax.imshow(grid,cmap='YlGnBu',vmin=80,vmax=96,aspect='auto')
    ax.set_xticks(range(3),['0.01','0.1','1']);ax.set_xlabel('Regularization parameter C');ax.set_yticks(range(9),[f'L{l} / {pool}' for l in [6,9,12] for pool in ['mean','temporal3','mean_std']],fontsize=7)
    for i in range(9):
        for j in range(3):ax.text(j,i,f'{grid[i,j]:.2f}',ha='center',va='center',color='white' if grid[i,j]>91 else 'black',fontsize=7)
    save(fig,'validation_search')
    train=load(ROOT/'runs/multidataset/miracl8_cpu_v2/training.json')['history']
    fig,ax=plt.subplots(figsize=(3.5,2.5),layout='constrained')
    ax.plot([r['epoch'] for r in train],[r['train']['loss'] for r in train],'o-',ms=3,label='Training',color=blue)
    ax.plot([r['epoch'] for r in train],[r['valid']['loss'] for r in train],'s-',ms=3,label='Validation',color=teal)
    ax.axvline(11,ls=':',color='gray',label='Selected epoch');ax.set_xlabel('Scratch-training epoch');ax.set_ylabel('Recorded loss');ax.legend(fontsize=7);save(fig,'scratch_loss')
    fig,ax=plt.subplots(figsize=(3.5,2.5),layout='constrained')
    ax.bar(['Train','Validation','Test','Quarantined'],[704,240,192,48],color=[blue,teal,gray,'#b7b7b7']);ax.bar(['Train'],[16],bottom=[704],color='#c88168',label='Tracking exclusions')
    for i,x in enumerate([720,240,192,48]):ax.text(i,x+12,str(x),ha='center')
    ax.set_ylim(0,820);ax.set_ylabel('Word utterances');ax.legend(fontsize=7);save(fig,'split_counts')
    fig,ax=plt.subplots(figsize=(3.5,2.5),layout='constrained')
    ax.bar(['Accepted\ncorrect','Rejected\ncorrect','Rejected\nwrong'],[137,45,10],color=[teal,blue,'#c88168'])
    for i,x in enumerate([137,45,10]):ax.text(i,x+3,str(x),ha='center')
    ax.set_ylim(0,160);ax.set_ylabel('Known test clips');save(fig,'rejection_counts')
    fig,ax=plt.subplots(figsize=(3.5,2.5),layout='constrained');timings=[load(RUN/'preparation.json')['seconds'],load(RUN/'extraction.json')['seconds'],s['seconds']]
    ax.barh(['ROI preparation','Feature extraction','Head search'],timings,color=[blue,teal,gray]);ax.invert_yaxis()
    for i,x in enumerate(timings):ax.text(x+8,i,f'{x:.2f}',va='center',fontsize=8)
    ax.set_xlim(0,1050);ax.set_xlabel('Recorded stage wall time (s)');save(fig,'runtime')
    latency=load(ROOT/'.qa/expanded-word-throughput.json')['measurements']
    fig,ax=plt.subplots(figsize=(3.5,2.5),layout='constrained');xx=np.arange(3)
    ax.bar(xx-.17,[x['round_trip_ms'] for x in latency],width=.34,color=blue,label='Round trip')
    ax.bar(xx+.17,[x['server']['wall_ms'] for x in latency],width=.34,color=teal,label='Server')
    ax.set_xticks(xx,['Request 1','Request 2','Request 3']);ax.set_ylabel('LRW endpoint latency (ms)');ax.legend(fontsize=7);save(fig,'latency')
    def diagram(name,labels):
        fig,ax=plt.subplots(figsize=(3.5,3.6));ax.set_xlim(0,1);ax.set_ylim(0,len(labels));ax.axis('off')
        for i,label in enumerate(labels):
            y=len(labels)-i-.75
            ax.add_patch(FancyBboxPatch((.05,y),.9,.53,boxstyle='round,pad=.015',facecolor='#eef3f5',edgecolor=blue,lw=.8))
            ax.text(.5,y+.265,label,ha='center',va='center',fontsize=8)
            if i<len(labels)-1:ax.annotate('',(.5,y-.45),(.5,y-.04),arrowprops={'arrowstyle':'->','color':blue})
        save(fig,name)
    diagram('architecture',['640 × 480 source RGB frames','Measured eye-axis and mouth alignment','96 × 96 grayscale → 29 frames','88 × 88 crop; mean .421 / std .165','Frozen AV-HuBERT Base: layer 9','Temporal mean + std → 1,536 features','Train-only scaler → eight-way head'])
    diagram('runtime_flow',['User starts video-only webcam','MediaPipe landmarks on browser CPU','Activity gate and utterance buffer','Mode-specific preprocessing + HTTP','FastAPI: model readiness + CPU lock','Model distribution + uncertainty','UI confirmation and history\n(current regression documented)'])
    equations={
      'normalization':r'$x^{\prime}_{tij}=(x_{tij}/255-0.421)/0.165$',
      'pooling':r'$v=[\mathrm{mean}_t(H);\ \mathrm{std}_t(H)]\in\mathbb{R}^{1536}$',
      'head':r'$p=\mathrm{softmax}(W((v-\mu_v)/s_v)+b)$',
      'objective':r'$\mathcal{J}=-\sum_n a_{y_n}\log p_{n,y_n}+\lambda\|W\|_2^2$',
      'softmax':r'$p_k=\exp(z_k)/\sum_{j=1}^{K}\exp(z_j)$',
      'ctc':r'$\mathcal{L}_{CTC}=-\log\sum_{\pi\in\mathcal{B}^{-1}(Y)}\prod_t p(\pi_t\mid X)$',
      'ema':r'$\bar p_t=\alpha p_t+(1-\alpha)\bar p_{t-1},\quad\alpha=0.6$',
      'reject':r'$p_{(1)}\geq\tau\quad\wedge\quad p_{(1)}-p_{(2)}\geq m$',
      'wer':r'$\mathrm{WER}=(S+D+I)/N$',
      'accuracy':r'$A=\frac{1}{N}\sum_{n=1}^{N}\mathbf{1}[\hat y_n=y_n]$',
      'metrics':r'$P_k=\frac{TP_k}{TP_k+FP_k},\ R_k=\frac{TP_k}{TP_k+FN_k},\ F1_k=\frac{2P_kR_k}{P_k+R_k}$',
      'temporal':r'$X(u)=(1-\alpha)X_{\lfloor u\rfloor}+\alpha X_{\lceil u\rceil}$',
    }
    for name,eq in equations.items():
        fig=plt.figure(figsize=(3.5,.43));fig.text(.5,.5,eq,ha='center',va='center',fontsize=11 if name!='metrics' else 9)
        save(fig,'eq_'+name)
    return names

def dynamic_tables(e,s,matrices,summary):
    tables={}
    tables['scores']=(['Measure','Scratch','Transfer'],[['Correct / test','24 / 192','182 / 192'],['Top-1 (%)','12.50','94.79'],['Top-5 (%)','Not asserted','99.48'],['Macro F1','See matrix','0.94668'],['95% Wilson CI (%)',*(f'{100*a:.2f}–{100*b:.2f}' for a,b in [wilson(24,192),wilson(182,192)])]])
    tables['perclass']=(['Word','Correct','Prec.','Recall','F1'],[[w,f"{d['correct']}/24",f"{d['precision']:.3f}",f"{d['recall']:.3f}",f"{d['f1-score']:.3f}"] for w,d in e['per_word'].items()])
    tables['candidates']=(['Layer','Pool','C','Valid. %','Loss'],[[c['layer'],{'mean':'mean','temporal3':'3 bins','mean_std':'mean+sd'}[c['pooling']],f"{c['C']:g}",f"{100*c['validation_accuracy']:.2f}",f"{c['validation_loss_usable']:.4f}"] for c in s['candidates']])
    tables['paired']=(['Scratch / transfer','Correct','Wrong'],[['Correct','24','0'],['Wrong','158','10']])
    return tables

def parse(content,tables):
    pages=[]
    for section in content.split('<!-- PAGE -->'):
        if not section.strip():continue
        blocks=[];lines=section.strip().splitlines();i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line:i+=1;continue
            if line.startswith('# '):blocks.append(('h1',line[2:]));i+=1
            elif line.startswith('## '):blocks.append(('h2',line[3:]));i+=1
            elif line.startswith('@figure '):
                name,caption=line[8:].split('|',1);blocks.append(('figure',(name.strip(),caption.strip())));i+=1
            elif line.startswith('@equation '):blocks.append(('equation',line.split()[1]));i+=1
            elif line.startswith('@table '):blocks.append(('table',tables[line.split()[1]]));i+=1
            elif line.startswith('|'):
                data=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                    if not all(re.fullmatch(r'[:\- ]+',c) for c in cells):data.append(cells)
                    i+=1
                blocks.append(('table',(data[0],data[1:])))
            else:
                para=[line];i+=1
                while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','@','|')):para.append(lines[i].strip());i+=1
                blocks.append(('p',' '.join(para)))
        pages.append(blocks)
    assert len(pages)==20,len(pages)
    table_titles=['Task definitions and evidence requirements','Local corpus inventory','Preprocessing contract','Loaded AV-HuBERT parameter accounting','Scratch-model tensor geometry','Paired recognition metrics','Paired outcome contingency','Per-word precision, recall, and F1','Complete validation candidate search','CPU workload and environment','Inference endpoint tasks','Capabilities of separate recognition modes','Proposed follow-up experiments','Reproduction source entry points']
    roman=['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII','XIV']
    count=0
    for page in pages:
        annotated=[]
        for block in page:
            if block[0]=='table':
                annotated.append(('tablecaption',f'TABLE {roman[count]}. {table_titles[count]}'));count+=1
            annotated.append(block)
        page[:]=annotated
    assert count==len(table_titles),count
    equation_number=0
    for page in pages:
        for i,(kind,value) in enumerate(page):
            if kind=='equation':
                equation_number+=1
                page[i]=(kind,(value,equation_number))
    return pages

def rich(text):
    text=escape(str(text));text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    return text.replace('`','')

class PaperEquation(Flowable):
    """Use the rendered math at its actual font size; number at right margin."""
    def __init__(self,name,number,width):
        Flowable.__init__(self)
        self.image=Image(str(ASSETS/('eq_'+name+'.png')))
        self.image.drawWidth=min(width-28,self.image.imageWidth*72/260)
        self.image.drawHeight=self.image.drawWidth*self.image.imageHeight/self.image.imageWidth
        self.width=width;self.height=self.image.drawHeight+10;self.number=number
    def wrap(self,*args):return self.width,self.height
    def draw(self):
        self.image.drawOn(self.canv,(self.width-25-self.image.drawWidth)/2,5)
        self.canv.setFont('TimesNR',10)
        self.canv.drawRightString(self.width,self.height/2-3,f'({self.number})')

class PaperFigure(Flowable):
    """Keep every illustration attached to its caption in one column."""
    def __init__(self,name,caption,width,style):
        Flowable.__init__(self);self.image=Image(str(ASSETS/(name+'.png')))
        image_width,image_height=self.image.imageWidth,self.image.imageHeight
        self.image.drawWidth=width;self.image.drawHeight=width*image_height/image_width
        assert self.image.drawWidth==width
        self.caption=Paragraph(rich(caption),style);self.width=width
        self.caption_height=self.caption.wrap(width,10000)[1]
        self.height=self.image.drawHeight+self.caption_height+13
    def wrap(self,*args):return self.width,self.height
    def draw(self):
        self.caption.drawOn(self.canv,0,4)
        self.image.drawOn(self.canv,0,self.caption_height+10)

def build_pdf(pages):
    for n,f in [('TimesNR','times.ttf'),('TimesNR-Bold','timesbd.ttf'),('TimesNR-Italic','timesi.ttf'),('TimesNR-BoldItalic','timesbi.ttf')]:pdfmetrics.registerFont(TTFont(n,str(Path('C:/Windows/Fonts')/f)))
    pdfmetrics.registerFontFamily('TimesNR',normal='TimesNR',bold='TimesNR-Bold',italic='TimesNR-Italic',boldItalic='TimesNR-BoldItalic')
    W,H=612,792;left=43;gap=22;cw=(W-2*left-gap)/2;bottom=46;top=746
    c=Canvas(str(OUT/(NAME+'.pdf')),pagesize=(W,H),pageCompression=1)
    c.setTitle(TITLE);c.setAuthor('Ravikumar Gupta; Omkar Gupta; Prof. Umang Patel');c.setSubject('Twenty-page evidence-based visual word recognition research manuscript')
    layout=[]
    def flow(blocks,leading=11.75,space=3):
        st=ParagraphStyle('body',fontName='TimesNR',fontSize=10,leading=leading,alignment=4,spaceAfter=space,firstLineIndent=10,splitLongWords=True)
        h1=ParagraphStyle('heading',parent=st,fontName='TimesNR',fontSize=10,alignment=1,leading=12,firstLineIndent=0,spaceAfter=7,spaceBefore=7)
        h2=ParagraphStyle('subheading',parent=st,fontName='TimesNR-Italic',alignment=0,firstLineIndent=0,spaceBefore=6,spaceAfter=4)
        cap=ParagraphStyle('caption',parent=st,fontSize=8,leading=9.5,alignment=0,spaceAfter=7)
        cap.firstLineIndent=0
        tc=ParagraphStyle('tablecaption',parent=cap,alignment=1,fontName='TimesNR',spaceBefore=5,spaceAfter=3)
        items=[]
        abstract=False
        for kind,value in blocks:
            if kind=='h1' and value=='Abstract':abstract=True;continue
            if kind=='p' and (abstract or value.startswith('**Index Terms')):
                ast=ParagraphStyle('abstract',parent=st,fontName='TimesNR-Bold',fontSize=9,leading=10.5,firstLineIndent=0,spaceAfter=7)
                items.append(Paragraph(('<b>Abstract—</b>' if abstract else '')+rich(value),ast));abstract=False
            elif kind in ['p','h1','h2','tablecaption']:
                displayed=value.upper() if kind=='h1' else value
                items.append(Paragraph(rich(displayed),{'p':st,'h1':h1,'h2':h2,'tablecaption':tc}[kind]))
            elif kind=='figure':items.append(PaperFigure(value[0],value[1],cw,cap))
            elif kind=='equation':items.append(PaperEquation(value[0],value[1],cw))
            elif kind=='table':
                headers,data=value;n=len(headers)
                ts=ParagraphStyle('cell',parent=st,fontSize=8,leading=9.5,alignment=0,firstLineIndent=0,spaceAfter=0)
                vals=[[Paragraph(rich(x),ts) for x in rr] for rr in [headers]+data]
                table=Table(vals,colWidths=[cw/n]*n,hAlign='LEFT',repeatRows=1)
                table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LINEABOVE',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]));table.spaceAfter=9;items.append(table)
        return items
    def height(f):return f.wrap(cw,10000)[1]+getattr(f,'spaceBefore',0)+getattr(f,'spaceAfter',0)
    for page,blocks in enumerate(pages,1):
        ytop=top
        if page==1:
            title=Paragraph(rich(TITLE),ParagraphStyle('title',fontName='TimesNR',fontSize=23,leading=26,alignment=1))
            hh=title.wrap(W-2*left,200)[1];title.drawOn(c,left,ytop-hh);ytop-=hh+12
            for text,size in [(AUTHORS,11)]+[(x,9) for x in AFFILIATIONS]:
                pp=Paragraph(rich(text),ParagraphStyle('author',fontName='TimesNR',fontSize=size,leading=size+2,alignment=1))
                hh=pp.wrap(W-2*left,100)[1];pp.drawOn(c,left,ytop-hh);ytop-=hh+3
            ytop-=12
        available=ytop-bottom
        def arrange(cap):
            items=flow(blocks);placements=[];ys=[]
            for col in range(2):
                y=ytop;floor=ytop-cap if col==0 else bottom;x=left+col*(cw+gap)
                while items:
                    f=items[0];before=getattr(f,'spaceBefore',0);after=getattr(f,'spaceAfter',0);hh=f.wrap(cw,10000)[1]
                    heading=isinstance(f,Paragraph) and f.style.name in ['heading','subheading','tablecaption']
                    extra=24 if heading and len(items)>1 else 0
                    if y-before-hh-after-extra>=floor:
                        items.pop(0);y-=before+hh;placements.append((f,x,y));y-=after
                    else:
                        remaining=y-before-floor-after
                        split=f.split(cw,remaining) if remaining>24 and not heading else []
                        if split and len(split)>1:
                            items[0:1]=split;continue
                        break
                ys.append(y)
            return items,placements,ys
        candidates=[]
        for cap in np.linspace(available*.55,available,46):
            rest,placed,ys=arrange(float(cap))
            if not rest:candidates.append((abs(ys[0]-ys[1]),placed,ys))
        if not candidates:
            rest,_,_=arrange(available)
            raise RuntimeError(f'Page {page} overflows; remaining {sum(height(x) for x in rest):.1f} pt')
        _,placed,ys=min(candidates,key=lambda x:x[0])
        for f,x,y in placed:f.drawOn(c,x,y)
        layout.append({'page':page,'column_occupancy':[(ytop-y)/available for y in ys],'characters':sum(len(str(v)) for k,v in blocks),'body_font_pt':10})
        c.showPage()
    c.save();return layout

def build_docx(pages):
    d=Document();sec=d.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=Inches(.6);sec.bottom_margin=Inches(.6);sec.left_margin=sec.right_margin=Inches(.6)
    normal=d.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10)
    normal.paragraph_format.space_after=Pt(3);normal.paragraph_format.line_spacing=Pt(11.75)
    normal.paragraph_format.first_line_indent=Pt(10);normal.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    for name in ['Title','Heading 1','Heading 2','Caption']:
        style=d.styles[name];style.font.name='Times New Roman';style.font.color.rgb=RGBColor(0,0,0)
        style.paragraph_format.first_line_indent=Pt(0)
    d.styles['Title'].font.size=Pt(23);d.styles['Title'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
    d.styles['Heading 1'].font.size=Pt(10);d.styles['Heading 1'].font.bold=False
    d.styles['Heading 1'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
    d.styles['Heading 2'].font.size=Pt(10);d.styles['Heading 2'].font.bold=False;d.styles['Heading 2'].font.italic=True
    d.styles['Caption'].font.size=Pt(8)
    d.add_heading(TITLE,0)
    for text,size in [(AUTHORS,11)]+[(a,9) for a in AFFILIATIONS]:
        para=d.add_paragraph(text);para.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.first_line_indent=Pt(0)
        for run in para.runs:run.font.size=Pt(size)
    sec=d.add_section(WD_SECTION_START.CONTINUOUS);cols=sec._sectPr.find(qn('w:cols'));cols.set(qn('w:num'),'2');cols.set(qn('w:space'),'360')
    for i,blocks in enumerate(pages):
        if i:d.add_page_break()
        abstract=False
        for kind,val in blocks:
            if kind=='h1' and val=='Abstract':abstract=True;continue
            if kind in ['h1','h2']:d.add_heading(val.upper() if kind=='h1' else val,1 if kind=='h1' else 2)
            elif kind=='p':
                para=d.add_paragraph(('Abstract—' if abstract else '')+val.replace('**',''))
                if abstract or val.startswith('**Index Terms'):
                    para.paragraph_format.first_line_indent=Pt(0)
                    for run in para.runs:run.bold=True;run.font.size=Pt(9)
                    abstract=False
            elif kind=='tablecaption':
                para=d.add_paragraph(val,style='Caption');para.alignment=WD_ALIGN_PARAGRAPH.CENTER
            elif kind in ['figure','equation']:
                name,caption=val if kind=='figure' else ('eq_'+val[0],'')
                im=Image(str(ASSETS/(name+'.png')))
                width=Inches(3.4) if kind=='figure' else Pt(min(224,im.imageWidth*72/260))
                d.add_picture(str(ASSETS/(name+'.png')),width=width)
                para=d.paragraphs[-1];para.paragraph_format.first_line_indent=Pt(0);para.alignment=WD_ALIGN_PARAGRAPH.CENTER
                if kind=='equation':
                    para=d.add_paragraph(f'({val[1]})',style='Caption');para.alignment=WD_ALIGN_PARAGRAPH.RIGHT
                if caption:d.add_paragraph(caption,style='Caption')
            elif kind=='table':
                headers,rr=val;t=d.add_table(rows=1,cols=len(headers));t.style='Light Shading Accent 1'
                for cell,txt in zip(t.rows[0].cells,headers):cell.text=str(txt)
                for row in rr:
                    for cell,txt in zip(t.add_row().cells,row):cell.text=str(txt)
                for row in t.rows:
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            para.paragraph_format.first_line_indent=Pt(0);para.paragraph_format.line_spacing=Pt(9.5)
                            for run in para.runs:run.font.size=Pt(8)
    d.core_properties.title=TITLE;d.core_properties.author='Ravikumar Gupta; Omkar Gupta; Prof. Umang Patel'
    d.save(OUT/(NAME+'.docx'))

def build_markdown(pages):
    output=['# '+TITLE,AUTHORS,*AFFILIATIONS,'']
    for i,blocks in enumerate(pages,1):
        output.append(f'<!-- PDF page {i} -->')
        for kind,value in blocks:
            if kind in ['h1','h2']:output.append(('## ' if kind=='h1' else '### ')+value)
            elif kind=='p':output.append(value)
            elif kind=='tablecaption':output.append('**'+value+'**')
            elif kind in ['figure','equation']:
                name,caption=value if kind=='figure' else ('eq_'+value[0],f'Equation ({value[1]})')
                output.append(f'![{caption}](figures/{name}.png)\n\n{caption}' if kind=='figure' else f'![{caption}](figures/{name}.png)')
            elif kind=='table':
                headers,rr=value
                output.extend(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |'])
                output.extend('| '+' | '.join(map(str,row))+' |' for row in rr)
            output.append('')
    (OUT/(NAME+'.md')).write_text('\n'.join(output),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(exist_ok=True)
    e,s,p,protocol,old,new,matrices,summary=evidence()
    fn=figures(e,s,p,protocol,old,new,matrices,summary)
    pages=parse(SOURCE.read_text(encoding='utf-8'),dynamic_tables(e,s,matrices,summary))
    layout=build_pdf(pages);build_docx(pages);build_markdown(pages)
    pdf=PdfReader(OUT/(NAME+'.pdf'));assert len(pdf.pages)==20
    texts=[p.extract_text() for p in pdf.pages];alltext='\n'.join(texts)
    for required in ['Ravikumar Gupta','Omkar Gupta','Umang Patel','94.79','12.50','182','192','13.33']:
        assert required in alltext,required
    for removed in ['Undergraduate Researchers','Assistant Professor and Project Supervisor','GUPTA, GUPTA, AND PATEL: CPU VISUAL WORD RECOGNITION','Author-prepared IEEE-style extended manuscript; not an IEEE publication.']:
        assert removed not in alltext,removed
    for removed in ['Data and Code Availability Statement','Reproduction Summary','The evidence inspected for this manuscript resides','The fixed PDF is accompanied','This extended author manuscript follows','No funding source, ethics approval number']:
        assert removed.lower() not in alltext.lower(),removed
    assert alltext.index('ACKNOWLEDGMENT')<alltext.rindex('REFERENCES')
    assert 'express their gratitude to Prof. Umang Patel' in ' '.join(alltext.split())
    assert all(f'({i})' in alltext for i in range(1,13))
    assert all(len(t)>1300 for t in texts),[len(t) for t in texts]
    receipt={'pages':20,'words_extracted':len(alltext.split()),'figures_excluding_equations':len([x for x in fn if not x.startswith('eq_')]),'equations':len([x for x in fn if x.startswith('eq_')]),'per_page':layout,'source_sha256':sha(SOURCE),'pdf_sha256':sha(OUT/(NAME+'.pdf')),'no_training_or_checkpoint_changes':True}
    (OUT/'document_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text('# IEEE-style extended research manuscript\n\nThe PDF contains exactly 20 pages. The DOCX is editable and may repaginate in Word. All graphs are supplied as PNG, SVG, and PDF. The manuscript is author-prepared, not accepted or published by IEEE. A venue-specific template and page limit must be applied before submission.\n\nRebuild from repository root:\n\n    .\\.venv\\Scripts\\python.exe scripts/build_ieee_evidence_paper.py\n\nSource: `submission_package/source/IEEE_Research_Paper_20_Pages.md`. `evidence_validation.json` verifies the 192 paired IDs, counts, confusion matrix and checkpoint hashes. `document_validation.json` records PDF pagination and layout. No new model was trained.\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
