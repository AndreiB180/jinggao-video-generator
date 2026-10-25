import {ExternalLink, Image as ImageIcon} from 'lucide-react';

const PROMPT = '这个是我的自媒体选题标题，需要展示出主题文字钩子，这个封面是用在百家号这个平台的，整体风格喜好要对应这个平台的人群';
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
  const model=values.cover_model||'gpt-image-2';
  return <section><h3><ImageIcon size={17}/>封面</h3>
    <p>成片保存到文件夹时，旁边会自动存封面图片（和视频同名，后面加“-封面-横版 / -竖版”）。</p>
    <label>封面方式<select value={mode} onChange={e=>change('cover_mode',e.target.value)}>
      <option value="design">排版封面（推荐）：自动挑视频里最好的画面，配百家号风格大字，约 ¥0.001 / 条</option>
      <option value="ai">AI 生图（GrsAI）：AI 重新画整张封面，约 ¥0.02–0.1 / 张，一两分钟</option>
      <option value="off">不生成封面</option>
    </select></label>
    <label>尺寸<select value={values.cover_sizes||'both'} onChange={e=>change('cover_sizes',e.target.value)}>
      <option value="both">横版 16:9 + 竖版 9:16</option>
      <option value="landscape">只要横版 16:9</option>
      <option value="portrait">只要竖版 9:16</option>
    </select></label>
    {mode==='ai'&&<>
      <label>GrsAI API Key<input type="password" autoComplete="off" value={values.cover_api_key||''}
        placeholder={values.cover_api_key_set?'已保存，留空保留原密钥':'填写 API Key'} onChange={e=>change('cover_api_key',e.target.value)}/></label>
      <div className="toolbar-actions"><a className="secondary small" href="https://grsai.ai" target="_blank" rel="noreferrer"><ExternalLink size={14}/>注册并获取 API Key</a></div>
      <label>模型<select value={models.some(m=>m.key===model)?model:'custom'} onChange={e=>e.target.value!=='custom'&&change('cover_model',e.target.value)}>
        {models.map(m=><option key={m.key} value={m.key}>{m.label}</option>)}
        {!models.some(m=>m.key===model)&&<option value="custom">{model}</option>}
      </select></label>
      <label>横版提示词<textarea rows={3} value={values.cover_prompt_landscape||''} placeholder={PROMPT} onChange={e=>change('cover_prompt_landscape',e.target.value)}/></label>
      <label>竖版提示词<textarea rows={3} value={values.cover_prompt_portrait||''} placeholder={PROMPT} onChange={e=>change('cover_prompt_portrait',e.target.value)}/></label>
      <p>留空就用灰色的默认提示词。标题会自动接在提示词后面；想放在中间，就在提示词里写 {'{标题}'}。标题优先用稿子里“标题：”那一行，没有就用第一句话。哪一张 AI 没画出来，就自动改用排版封面。</p>
      <label className="checkbox"><input type="checkbox" checked={(values.cover_reference||'1')==='1'} onChange={e=>change('cover_reference',e.target.checked?'1':'0')}/>把视频开头最清楚的一帧作为参考图交给 AI（人物、场景更贴视频）</label>
      <label>接口地址<input value={values.cover_base_url||''} placeholder="https://grsai.dakka.com.cn" onChange={e=>change('cover_base_url',e.target.value)}/></label>
    </>}
  </section>;
}
