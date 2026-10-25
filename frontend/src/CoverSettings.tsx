import {ExternalLink, Image as ImageIcon} from 'lucide-react';

const PROMPT = '这个是我的自媒体选题标题，需要展示出主题文字钩子，这个封面是用在百家号这个平台的，整体风格喜好要对应这个平台的人群';
const BACKGROUND = '为这期视频画一张百家号封面的背景图。画面里不要出现任何文字、字母、数字、标志和水印。主体人物或场景放在画面右侧，左侧留出较暗、干净的区域放标题。写实、电影感、冲击力强，符合百家号读者的喜好。';
const models = [
  {key:'gpt-image-2', label:'gpt-image-2 · 约 ¥0.03–0.06 / 张'},
  {key:'nano-banana-2-lite', label:'nano-banana-2-lite · 约 ¥0.02 / 张（最便宜，中文字一般）'},
  {key:'gpt-image-2-vip', label:'gpt-image-2-vip · 约 ¥0.1 / 张，画质更高'},
  {key:'nano-banana-2', label:'nano-banana-2 · 约 ¥0.06 / 张'},
  {key:'nano-banana-pro', label:'nano-banana-pro · 约 ¥0.09 / 张'},
];

/** Covers saved next to each film: drawn by an image model (GrsAI) or a frame with the title. */
export function CoverSettings({values,change}:{values:Record<string,any>;change:(key:string,value:string)=>void}) {
  const mode=values.cover_mode||'design';
  const ai=mode==='ai'||mode==='ai_bg';
  const model=values.cover_model||(mode==='ai_bg'?'nano-banana-2-lite':'gpt-image-2');
  return <section><h3><ImageIcon size={17}/>封面</h3>
    <p>成片保存到文件夹时，旁边会自动存封面图片（和视频同名，后面加“-封面-横版 / -竖版”）。</p>
    <label>封面方式<select value={mode} onChange={e=>change('cover_mode',e.target.value)}>
      <option value="design">排版封面：视频里最好的画面 + 百家号风格大字，约 ¥0.001 / 条</option>
      <option value="ai_bg">AI 画背景 + 排版大字（推荐）：AI 画一张不带字的画面，字由软件排，约 ¥0.02 / 条</option>
      <option value="ai">AI 整张生成：标题也由 AI 写在图上，约 ¥0.06–0.12 / 条（横竖各一张）</option>
      <option value="off">不生成封面</option>
    </select></label>
    <label>尺寸<select value={values.cover_sizes||'both'} onChange={e=>change('cover_sizes',e.target.value)}>
      <option value="both">横版 16:9 + 竖版 9:16</option>
      <option value="landscape">只要横版 16:9</option>
      <option value="portrait">只要竖版 9:16</option>
    </select></label>
    {ai&&<>
      <p>AI 封面用 GrsAI 的接口：在下面的链接注册，充值（10 元起），创建 API Key 填到这里。AI 在后台画，一两分钟一条，不耽误下一条视频；哪一张没画出来，就自动改用排版封面。</p>
      <label>GrsAI API Key<input type="password" autoComplete="off" value={values.cover_api_key||''}
        placeholder={values.cover_api_key_set?'已保存，留空保留原密钥':'填写 API Key'} onChange={e=>change('cover_api_key',e.target.value)}/></label>
      <div className="toolbar-actions"><a className="secondary small" href="https://grsai.ai" target="_blank" rel="noreferrer"><ExternalLink size={14}/>注册并获取 API Key</a></div>
      <label>模型<select value={models.some(m=>m.key===model)?model:'custom'} onChange={e=>e.target.value!=='custom'&&change('cover_model',e.target.value)}>
        {models.map(m=><option key={m.key} value={m.key}>{m.label}</option>)}
        {!models.some(m=>m.key===model)&&<option value="custom">{model}</option>}
      </select></label>
      {mode==='ai_bg'&&<label>背景提示词<textarea rows={3} value={values.cover_prompt_background||''} placeholder={BACKGROUND} onChange={e=>change('cover_prompt_background',e.target.value)}/></label>}
      {mode==='ai'&&<>
        <label>横版提示词<textarea rows={3} value={values.cover_prompt_landscape||''} placeholder={PROMPT} onChange={e=>change('cover_prompt_landscape',e.target.value)}/></label>
        <label>竖版提示词<textarea rows={3} value={values.cover_prompt_portrait||''} placeholder={PROMPT} onChange={e=>change('cover_prompt_portrait',e.target.value)}/></label>
      </>}
      <p>留空就用灰色的默认提示词。视频的标题会自动接在提示词后面{mode==='ai'?<>（想放在中间，就在提示词里写 {'{标题}'}）</>:''}。标题优先用稿子里“标题：”那一行，没有就用第一句话。</p>
      <label className="checkbox"><input type="checkbox" checked={(values.cover_reference||'1')==='1'} onChange={e=>change('cover_reference',e.target.checked?'1':'0')}/>把视频开头最清楚的一帧作为参考图交给 AI（人物、场景更贴视频）</label>
      <label>接口地址<input value={values.cover_base_url||''} placeholder="https://grsai.dakka.com.cn" onChange={e=>change('cover_base_url',e.target.value)}/></label>
    </>}
  </section>;
}
