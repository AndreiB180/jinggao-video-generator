import {useEffect, useState} from 'react';

type Totals = {requests:number; estimated_cny:number; reserved_cny:number; total_cny:number; limit_cny:number; unresolved_requests:number;cost_complete?:boolean;unpriced_requests?:number};
export type ModelCostReport = {day:string; daily:Totals; project:Totals|null; provider?:Omit<Totals,'limit_cny'>&{host:string;limit_cny:number|null}; first_recorded_at:number|null; note:string; pricing_url:string; pricing_date:string; categories:{stage:string;category:string;model:string;requests:number;total_cny:number}[];tasks?:{stage:string;category:string;task:string;model:string;requests:number;total_cny:number|null;pricing_status:string}[]};
const money = (value:number) => `¥${value.toFixed(4)}`;

export function ModelCostBar({report, settings}:{report?:ModelCostReport;settings:()=>void}) {
  if (!report?.project) return null;
  const p = report.project, d = report.daily;
  return <div className="model-cost-bar">
    <span>已记费用 · 本项目 {money(p.estimated_cny)}{p.reserved_cny>0&&` + 预留 ${money(p.reserved_cny)}`} / ¥{p.limit_cny.toFixed(2)}</span>
    {p.cost_complete===false&&<span role="status">另有 {p.unpriced_requests} 次配音费用未核明，总费用待确认</span>}
    <span>今日累计含预留 {money(d.total_cny)} / ¥{d.limit_cny.toFixed(2)}</span>
    {report.provider?.limit_cny!=null&&<span>当前服务累计 {money(report.provider.total_cny)} / ¥{report.provider.limit_cny.toFixed(2)}</span>}
    <button className="text-link" onClick={settings}>费用与预算</button>
  </div>;
}

export function CostSettings({values, change}:{values:Record<string,any>;change:(key:string,value:string)=>void}) {
  const [report,setReport] = useState<ModelCostReport|null>(null), [error,setError] = useState('');
  useEffect(()=>{
    let mounted = true;
    const refresh = async()=>{
      try {
        const response = await fetch('/api/costs');
        if(!response.ok) throw new Error('费用账本暂时无法读取');
        const data = await response.json();
        if(mounted) {setReport(data);setError('')}
      } catch(e) {if(mounted) setError((e as Error).message)}
    };
    refresh(); const timer = setInterval(refresh, 3000);
    return ()=>{mounted=false;clearInterval(timer)};
  },[]);
  const field = (key:string,label:string,step='0.01')=><label key={key}>{label}<input type="number" min="0" step={step} value={values[key]??''} onChange={e=>change(key,e.target.value)}/></label>;
  const category:Record<string,string> = {text:'旧版文字请求',vision:'旧版视觉请求',planning:'分镜与检索词',requirements:'原稿要求提取',evidence_review:'文字证据核验',visual_review:'实际画面核验',candidate_ranking:'候选预览筛选',pixel_observation:'画面描述',pixel_quality:'画质与画面类型',library_ranking:'库内素材排序',scene_ranking:'来源片段排序',retrieval_grouping:'检索分组',entity_aliases:'人物来源别名',privacy_scan:'联系方式遮挡检查',speech:'配音',tts:'配音'};
  const grouped = new Map<string,{category:string;model:string;requests:number;total_cny:number;unpriced:boolean}>();
  const entries = report?.tasks?.map(c=>({...c,category:c.task})) || report?.categories || [];
  for(const c of entries) {
    const task = c.category.startsWith('tts_')?'tts':c.category;
    const key = task+'|'+c.model, old = grouped.get(key);
    grouped.set(key,{...c,category:task,requests:c.requests+(old?.requests||0),total_cny:(c.total_cny||0)+(old?.total_cny||0),unpriced:c.total_cny===null||Boolean(old?.unpriced)});
  }
  return <section>
    <h3>生成费用与预算</h3>
    <div className="field-grid">{field('model_job_budget_cny','单次任务上限（元）')}{field('model_project_budget_cny','每项目累计上限（元）')}{field('model_daily_budget_cny','每日累计上限（元）')}{field('model_provider_budget_cny','当前服务累计上限（元，可留空）')}</div>
    <p>发请求前预留费用，超限自动停止并保存进度。项目费用包含历次重试，更换密钥也不清零；0 表示暂停付费模型调用。每日按本机日期计算。</p>
    <p>服务累计按接口域名统计，跨项目、跨日和更换密钥均不清零。留空仅关闭服务累计上限，项目与每日上限仍生效。</p>
    {report&&<>
      <div className="inline-note">今日已记账 {money(report.daily.estimated_cny)}，进行中或用量未知预留 {money(report.daily.reserved_cny)}，共 {report.daily.requests} 次记录。{!report.first_recorded_at&&'账本启用后尚无请求。'}</div>
      {report.daily.cost_complete===false&&<p role="status">另有 {report.daily.unpriced_requests} 次配音未核明收费。以上金额只是已记费用，尚不能据此判断整片是否低于预算。</p>}
      {report.provider&&<div className="inline-note">{report.provider.host} · 累计含预留 {money(report.provider.total_cny)}{report.provider.limit_cny!=null&&` / ¥${report.provider.limit_cny.toFixed(2)}`} · {report.provider.requests} 次记录</div>}
      {grouped.size>0&&<ul className="cost-breakdown">{Array.from(grouped.entries()).map(([key,c])=><li key={key}>{category[c.category]||c.category} · {c.model}：{c.requests} 次 · {c.unpriced?'包含未核明费用':`含预留 ${money(c.total_cny)}`}</li>)}</ul>}
      <p>{report.note} <a href={report.pricing_url} target="_blank" rel="noreferrer">官方价格</a>（核对日期 {report.pricing_date}）。</p>
    </>}
    {error&&<p role="alert">{error}</p>}
    <details><summary>其他模型或自定义单价</summary><p>官方 DeepSeek Flash / V4 Pro、阿里云北京 Qwen3.7 Flash 留空即可；Qwen 按请求输入长度选择阶梯价格。其他接口须填写以下三项人民币单价；文稿和视觉模型不同时填较高单价。费用按填写值估算。</p>
      <div className="field-grid">{field('cost_input_per_million','输入 · 元/百万 tokens','0.0001')}{field('cost_cached_per_million','缓存输入 · 元/百万 tokens','0.0001')}{field('cost_output_per_million','输出 · 元/百万 tokens','0.0001')}{field('cost_image_tokens','每张图片预留 tokens','1')}</div>
      <p>自定义视觉模型还须填写每图 token 上限；未知价格或图片计费规则时停止调用。DeepSeek Flash 每图按 1024 tokens 加消息开销预留；Qwen3.7 Flash 默认每图按 2560 tokens 加消息开销预留。</p>
    </details>
  </section>;
}
