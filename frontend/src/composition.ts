import type {Project} from './types';

// Match curation.composition: representation changes the video target numerator,
// never the playable duration, image movement or the unrestricted mixed policy.
export function composition(p:Project) {
 let total=0,video=0,images=0,missing=0,frozen=0,illustration=0,unclassified=0,legacy=0;
 const unresolved:number[]=[],short:number[]=[],low:number[]=[],illustrationShots:number[]=[],unclassifiedShots:number[]=[];
 for(const [i,s] of p.shots.entries()) {
  const index=i+1,a=p.assets.find(a=>a.id===s.selected_asset),duration=p.full_audio?s.duration:(s.audio?.duration||s.duration);
  total+=duration;
  if(a?.kind==='video') {
   const playable=Math.max(0,(a.duration||0)-(s.clip_start||0)),used=Math.min(playable,duration),quality=s.verification?.visual_quality;
   if(quality==null){video+=used;legacy+=used;}
   else if(quality.representation==='animation'||quality.representation==='mixed'){illustration+=used;illustrationShots.push(index);}
   else if(quality.representation==='live_action'||quality.representation==='screen_capture'||quality.representation==='recorded_mix')video+=used;
   else {unclassified+=used;unclassifiedShots.push(index);}
   frozen+=Math.max(0,duration-playable);
   if(duration-playable>.16)short.push(index);
   if(Math.min(a.width||0,a.height||0)<(p.min_video_height??720))low.push(index);
  } else if(a?.kind==='image')images+=duration;
  else missing+=duration;
  const named=!!s.persons?.length||['person','event'].includes(s.evidence_mode||'')||!!(s.event&&s.event_scope!=='background');
  let manual=false;
  if((s.source_review_note||'').trim().length>=8){try{const u=new URL(s.source_review_url||'');manual=['http:','https:'].includes(u.protocol)&&!!u.hostname&&!u.username&&!u.password;}catch{/* An invalid source URL is not evidence. */}}
  if(named&&!((s.verification?.identity_evidence?.status==='supported'&&s.verification?.selection_blocked===false)||manual))unresolved.push(index);
 }
 const policy=p.media_policy??'mixed',ratio=total?video/total:0,target=policy==='video_only'?1:(p.min_video_ratio??.8);
 const round=(value:number,digits=3)=>Number(value.toFixed(digits));
 return {total_seconds:round(total),video_seconds:round(video),image_seconds:round(images),missing_seconds:round(missing),
  frozen_seconds:round(frozen),illustration_video_seconds:round(illustration),unclassified_video_seconds:round(unclassified),legacy_video_seconds:round(legacy),
  video_ratio:round(ratio,4),target_video_ratio:target,meets_target:policy==='mixed'||ratio+.004>=target,
  identity_pending:unresolved,short_video_shots:short,low_resolution_shots:low,illustration_video_shots:illustrationShots,unclassified_video_shots:unclassifiedShots};
}
