export function ModelRouting({values, change}:{values:Record<string,any>;change:(key:string,value:string)=>void}) {
 const routes = [
  ['planning_model','分镜与检索词','只规划画面；原稿切分在本地完成'],
  ['requirements_model','原稿要求提取','提取人物、事件和必须出现的画面'],
  ['review_model','文字证据核验','对照已缓存的画面描述与原稿'],
 ];
 return <section><h3>按环节选择模型</h3>
  <p>能在本地完成的环节不调用付费模型。以下留空时使用上方文稿模型；画面观察与质检使用视觉模型。</p>
  <div className="field-grid">{routes.map(([key,label,note])=><label key={key}>{label}<input value={values[key]||''} placeholder={values.model_name||'使用文稿模型'} onChange={e=>change(key,e.target.value)}/><small>{note}</small></label>)}</div>
  <p>字幕断句与音轨对齐：本地语音模型；素材去重、视频合成：本地工具。均无模型接口费。配音按所选服务单独计费，已有配音直接复用。</p>
  <p>北京 Qwen3.7 Flash 短输入单价较低；Qwen Turbo 仅输出单价更低，输入单价反而更高。不会自动切换到更贵模型，也不会自动补买额度。</p>
 </section>;
}
