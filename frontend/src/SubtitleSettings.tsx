import {useEffect, useState} from 'react';
import {Type} from 'lucide-react';
import {captionCss, useSubtitleFont} from './ShotPreview';

type Option = {key:string;label:string;file?:string;available?:boolean;css?:string;em?:number};
type Styles = {fonts:Option[];sizes:Option[];colours:Option[]};

/** Burned-in subtitle look: font, size and colour, with a live sample at the export's proportions. */
export function SubtitleSettings({values,change,compact=false}:{values:Record<string,any>;change:(key:string,value:string)=>void;compact?:boolean}) {
 const [styles,setStyles]=useState<Styles|null>(null);
 useEffect(()=>{fetch('/api/subtitle-styles').then(r=>r.json()).then(setStyles).catch(()=>setStyles(null))},[]);
 const font=values.subtitle_font||'black', size=values.subtitle_size||'84', colour=values.subtitle_color||'white';
 const chosen=styles?.fonts.find(f=>f.key===font);
 const family=useSubtitleFont(chosen?.available?chosen.file:undefined);
 const px=Number(size)/1080;
 const sample={font,font_file:chosen?.file||'',colour:styles?.colours.find(c=>c.key===colour)?.css||'#fff',
  size:px*(chosen?.em||.69),outline:Math.max(2,Number(size)/11)/1080,margin:.07};
 const colourSelect=<label>颜色<select value={colour} onChange={e=>change('subtitle_color',e.target.value)}>{(styles?.colours||[{key:'white',label:'白字黑边'}]).map(c=><option key={c.key} value={c.key}>{c.label}</option>)}</select></label>;
 return <section className={compact?'subtitle-compact':''}>{!compact&&<><h3><Type size={17}/>字幕样式</h3>
  <p>成片字幕每次只显示一行；口播超过一行的句子会在分镜时自动拆成多个镜头。修改后对之后的导出生效。</p></>}
  <div className="subtitle-sample"><span style={captionCss(sample,family,16/9)}>范仲淹为什么要抬高米价</span></div>
  <div className={compact?'field-grid three':'field-grid'}>
   <label>字体<select value={font} onChange={e=>change('subtitle_font',e.target.value)}>{(styles?.fonts||[{key:'black',label:'思源黑体 粗'}]).map(f=><option key={f.key} value={f.key} disabled={f.available===false}>{f.label}{f.available===false?'（未下载）':''}</option>)}</select></label>
   <label>字号<select value={size} onChange={e=>change('subtitle_size',e.target.value)}>{(styles?.sizes||[{key:'84',label:'大（默认）'}]).map(s=><option key={s.key} value={s.key}>{s.label}</option>)}</select></label>
   {compact&&colourSelect}
  </div>
  {!compact&&colourSelect}
 </section>;
}
