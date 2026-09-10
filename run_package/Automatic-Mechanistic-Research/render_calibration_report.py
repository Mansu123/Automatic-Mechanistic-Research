#!/usr/bin/env python3
"""Render actual engineering measurements; never substitute for full-model reports."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def render(result_path,acceptance_path,outdir):
    result_path=Path(result_path);r=json.loads(result_path.read_text())
    a=json.loads(Path(acceptance_path).read_text());out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    if r.get('scientific_status')!='pilot_only_not_eligible_for_main_results':
        raise ValueError('This renderer is restricted to engineering pilots')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':170,'axes.titleweight':'bold'})
    blue,orange,teal='#235789','#C8772D','#258B83'
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.png'),bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    data=r['records'];n=len(data);layers=r['model']['n_layers'];heads=r['model']['n_heads']
    fig,ax=plt.subplots(1,2,figsize=(9,2.7))
    acc=r['clean_pairwise_accuracy'];ax[0].barh(['IOI calibration'],[acc['mean']],color=blue)
    ax[0].errorbar([acc['mean']],[0],xerr=[[acc['mean']-acc['ci95'][0]],[acc['ci95'][1]-acc['mean']]],fmt='none',ecolor='black',capsize=4)
    ax[0].set_xlim(0,1.05);ax[0].set_xlabel('Pairwise accuracy (Wilson 95% interval)')
    ax[0].set_title(f'A. Accuracy: {int(acc["successes"])}/{n}')
    x=np.arange(n)
    ax[1].plot(x,[q['clean']['margin'] for q in data],'o-',color=blue,label='Clean')
    ax[1].plot(x,[q['corrupt']['margin'] for q in data],'s-',color=orange,label='Role-swapped')
    ax[1].axhline(0,color='grey',lw=.7);ax[1].set_xlabel('Independent calibration item');ax[1].set_ylabel('Full-answer log-probability margin');ax[1].legend(frameon=False);ax[1].set_title('B. Signed behavior contrast')
    save(fig,'01_capability_and_margin')
    fig,ax=plt.subplots(figsize=(9,3.1))
    for kind,color,label in [('residual',blue,'Whole residual (information availability)'),('block_update',teal,'Block update (computation intervention)'),('mlp',orange,'MLP output')]:
        vals=sorted([q for q in r['aggregate_effects'] if q['kind']==kind],key=lambda q:q['layer'])
        means=[q['recovery']['estimate'] if q['recovery']['estimate'] is not None else np.nan for q in vals]
        lows=[q['recovery']['ci95'][0] if q['recovery']['ci95'] else np.nan for q in vals]
        highs=[q['recovery']['ci95'][1] if q['recovery']['ci95'] else np.nan for q in vals]
        ax.plot(range(layers),means,'o-',ms=3,color=color,label=label);ax.fill_between(range(layers),lows,highs,color=color,alpha=.12)
    ax.axhline(0,color='grey',lw=.7);ax.axhline(1,color='grey',lw=.7,ls='--');ax.set_xlabel('Layer index (zero based)');ax.set_ylabel('Recovered fraction of mean clean-corrupted gap');ax.set_xticks(range(layers));ax.legend(frameon=False,fontsize=8);ax.set_title('C. Layer interventions at the last prompt position')
    save(fig,'02_layer_interventions')
    mat=np.full((layers,heads),np.nan)
    head_rows=[q for q in r['aggregate_effects'] if q['kind']=='head']
    for q in head_rows:
        if q['recovery']['estimate'] is not None:mat[q['layer'],q['head']]=q['recovery']['estimate']
    lim=max(.1,float(np.nanmax(np.abs(mat))))
    fig,ax=plt.subplots(figsize=(9,3.9))
    im=ax.imshow(mat,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto',origin='lower')
    ax.set_xticks(range(heads));ax.set_yticks(range(layers));ax.set_xlabel('Query head');ax.set_ylabel('Layer');ax.set_title('D. Exact head patch recovery at the last prompt position');fig.colorbar(im,ax=ax,label='Recovered fraction (signed, unclipped)')
    save(fig,'03_head_effects')
    ranked=sorted([q for q in head_rows if q['recovery']['estimate'] is not None],key=lambda q:abs(q['recovery']['estimate']),reverse=True)[:12]
    fig,ax=plt.subplots(figsize=(9,3.4))
    values=[q['recovery']['estimate'] for q in ranked]
    ax.barh(range(len(ranked)),values,color=[blue if v>=0 else orange for v in values])
    for i,q in enumerate(ranked):
        ci=q['recovery']['ci95']
        if ci:ax.plot(ci,[i,i],color='black',lw=.8)
    ax.set_yticks(range(len(ranked)),[f'L{q["layer"]}H{q["head"]}' for q in ranked]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.6);ax.set_xlabel('Recovered fraction and paired bootstrap 95% interval');ax.set_title('E. Largest observed head effects - exploratory ranking')
    save(fig,'04_ranked_effects')

    pdfmetrics.registerFont(TTFont('ReportSans',findfont('DejaVu Sans')))
    pdfmetrics.registerFont(TTFont('ReportBold',findfont('DejaVu Sans:weight=bold')))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Text',fontName='ReportSans',fontSize=9,leading=13,spaceAfter=8))
    styles.add(ParagraphStyle(name='TitleCustom',fontName='ReportBold',fontSize=22,leading=27,textColor=colors.HexColor(blue),spaceAfter=14))
    styles.add(ParagraphStyle(name='SectionCustom',fontName='ReportBold',fontSize=13,leading=17,spaceBefore=8,spaceAfter=9,textColor=colors.HexColor(blue)))
    story=[]
    def para(text,style='Text'):story.append(Paragraph(text,styles[style]))
    def chart(name):
        from PIL import Image as PILImage
        p=out/(name+'.png');w,h=PILImage.open(p).size
        story.append(Image(str(p),width=510,height=510*h/w));story.append(Spacer(1,8))
    def table(rows,widths):
        cells=[[Paragraph(escape(str(c)),styles['Text']) for c in row] for row in rows]
        t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EFF5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor(blue)),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#D4DDE4')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.append(t);story.append(Spacer(1,9))
    para('GPT-2<br/>Engineering calibration report','TitleCustom')
    para('<b>PILOT ONLY - formal 25-model evaluation has not started.</b> This report verifies repaired measurements on a real checkpoint. It covers eight IOI calibration items and makes no Confirmed-circuit, edge-recall, hierarchy-advantage, or novelty claim.')
    table([['Measurement','Observed value'],['Checkpoint',r['model']['model_id']+' @ '+r['configuration']['revision']],['Runtime',r['configuration']['device']+' / '+r['configuration']['dtype']+' / '+r['model']['attention_backend']],['Architecture',f'{layers} layers, {heads} query heads; {r["model"]["n_params"]:,} parameters'],['Calibration items',f'{n} distinct clusters from validation; sealed test/stress unused'],['Clean pairwise accuracy',f'{acc["mean"]:.1%}; Wilson 95% interval {acc["ci95"][0]:.1%} to {acc["ci95"][1]:.1%}'],['Real-checkpoint acceptance',f'{sum(c["passed"] for c in a["checks"])}/{len(a["checks"])} checks passed'],['Measured compute',f'{r["telemetry"]["model_forward_calls"]:,} forwards; {r["telemetry"]["wall_seconds"]:.2f} seconds; no API calls']], [145,365])
    chart('01_capability_and_margin')
    para('The clean and corrupted prompts swap the repeated name. The score uses every answer token. High accuracy on eight calibration items is a wiring result with substantial uncertainty, not a broad capability estimate.')
    story.append(PageBreak())
    para('Layer and head interventions','SectionCustom')
    chart('02_layer_interventions')
    para('Whole-residual recovery records where task information is available. Its final-layer value can approach one by construction. Block-update and MLP interventions change a narrower computation. Ratios use the mean signed effect divided by the mean clean-corrupted gap; values are not clipped.')
    chart('03_head_effects')
    para('All query heads are included at one explicit token position. These measurements neither reconstruct a complete circuit nor test every token position. Head effects are not published-graph edge recall.')
    story.append(PageBreak())
    para('Exploratory effects and acceptance evidence','SectionCustom')
    chart('04_ranked_effects')
    para('Ranking and effect estimation share this engineering calibration sample. Treat the ranking as exploratory. Formal circuit selection must use discovery data and be verified on untouched examples with random, sham, joint-ablation and minimality controls.')
    table([['Acceptance family','Outcome'],['Finite full-continuation scores','Passed'],['Self-patch identity','Passed for residual, block update, MLP and head at early/middle/final layers'],['Changed-prompt and joint ablation','Passed; baseline restored after hooks removed'],['Multi-token answers and gradients','Passed'],['Hook cleanup','Passed; no intervention hooks remained']], [170,340])
    para('Still required for the requested study','SectionCustom')
    para('GPU-server connection and API model configuration; gated-checkpoint access; complete reviewed datasets for 203 behaviors; remaining scientific repairs and method dependencies; full 25-model runs, ablations, independent judging and expert review. Missing results will remain missing, rather than being shown as zero or success.')
    para('Provenance','SectionCustom')
    para('Run ID:<br/><font size="7">'+r['run_id']+'</font><br/>Dataset SHA-256:<br/><font size="7">'+r['configuration']['dataset_hash']+'</font><br/>Raw measurements: '+escape(str(result_path.resolve())))
    def footer(canv,doc):
        canv.setFont('ReportSans',8);canv.setFillColor(colors.HexColor('#61717F'))
        canv.drawString(40,23,'AutoMechInterp | Engineering pilot only | 5 September 2026')
        canv.drawRightString(A4[0]-40,23,str(doc.page))
    pdf=out/'GPT2_ENGINEERING_PILOT.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=35,bottomMargin=40,title='GPT-2 engineering calibration - not formal evaluation',author='AutoMechInterp').build(story,onFirstPage=footer,onLaterPages=footer)
    print(pdf)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--acceptance',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();render(a.result,a.acceptance,a.output_dir)
