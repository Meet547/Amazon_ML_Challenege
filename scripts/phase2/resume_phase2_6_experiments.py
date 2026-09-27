"""Resumable fixed-sample Phase 2.6 experiments; all bulky caches live in /private/tmp."""
import hashlib, json, resource, time, os
from pathlib import Path
import polars as pl
from src.phase2.loader import load_entity_tables
from src.phase2.blocks import (GENERIC_ADDRESS_TOKENS, LEGAL_FORM_TOKENS, source2_and_source3,
    token_key_frame, address_token_key_frame, token_pair_key_frame, name_address_composite_key_frame,
    name_token_pair_key_frame, numeric_address_key_frame, profile_key_blocks)
from src.phase2.candidate_generator import (_source1, _key_pairs, _cap_token_frequency)
from src.phase2.deduplicate import deduplicate_with_provenance
from src.phase2.evaluate import evaluate_recall
from src.phase2.config import NAME_TOKEN_MIN_LENGTH
from src.evaluation.ground_truth import load_ground_truth

ROOT=Path.cwd(); CACHE=Path(os.environ.get('PHASE2_6_CACHE_DIR', '/private/tmp/phase2_6_cache')); CACHE.mkdir(parents=True, exist_ok=True)
MANIFEST=ROOT/'outputs/metrics/phase2_6_experiments.json'
IDS=(ROOT/'outputs/metrics/phase2_100k_sample_ids.txt').read_text().splitlines()
if len(IDS)!=100_000: raise RuntimeError('wrong frozen sample size')
tables=load_entity_tables('train'); full=_source1(tables); left=full.filter(pl.col('s1_id').is_in(IDS))
targets=source2_and_source3(tables).with_columns(pl.col('candidate_id').str.slice(0,2).alias('candidate_source'))
labels=load_ground_truth(ROOT/'datasets/train/train_ground_truth.tsv',ROOT/'datasets/train/train_source1.tsv',ROOT/'datasets/train/train_source2.tsv',ROOT/'datasets/train/train_source3.tsv').filter(pl.col('source1_entity_id').is_in(IDS))
truth=labels.lazy().select(pl.col('source1_entity_id').alias('s1_id'),pl.col('matched_entity_ids').alias('candidate_id')).explode('candidate_id',empty_as_null=True).filter(pl.col('candidate_id').is_not_null()).collect(engine='streaming').unique(['s1_id','candidate_id'])
sample=pl.DataFrame({'s1_id':IDS})
baseline_path=CACHE/'baseline.parquet'
if not baseline_path.exists(): raise RuntimeError('baseline cache missing')
BASE=pl.read_parquet(baseline_path)

def write_parquet_atomic(frame, path: Path, *, lazy=False):
    temp_path=path.with_name(path.stem+'.tmp.parquet')
    if lazy:
        frame.sink_parquet(temp_path,compression='zstd')
    else:
        frame.write_parquet(temp_path,compression='zstd')
    temp_path.replace(path)

def compute(name, comp, config, parent):
    tic=time.perf_counter(); pairfile=CACHE/(name+'.parquet'); status='complete'; failure=None
    try:
        if not pairfile.exists(): write_parquet_atomic(comp,pairfile)
        extra=pl.scan_parquet(pairfile)
        union=deduplicate_with_provenance(pl.concat([parent.lazy(),extra],how='vertical')).collect(engine='streaming')
        union_path=CACHE/(name+'_union.parquet'); write_parquet_atomic(union,union_path)
        # Compute recall directly from one materialized union; no repeated evaluate_recall replay.
        matched=truth.join(union.select('s1_id','candidate_id').unique(),on=['s1_id','candidate_id'],how='inner')
        src={}
        for prefix in ('S2-','S3-'):
            t=truth.filter(pl.col('candidate_id').str.starts_with(prefix)).height
            f=matched.filter(pl.col('candidate_id').str.starts_with(prefix)).height
            src[prefix[:2]]={'true_pairs':t,'recovered_pairs':f,'recall':f/t if t else None}
        tcount=truth.height; rec=matched.height
        tper=truth.group_by('s1_id').len().rename({'len':'tn'}); rper=matched.group_by('s1_id').len().rename({'len':'rn'})
        fulls=tper.join(rper,on='s1_id',how='left').with_columns(pl.col('rn').fill_null(0)).select((pl.col('rn')==pl.col('tn')).sum()).item()
        counts=union.group_by('s1_id').len().rename({'len':'n'})
        dist=sample.join(counts,on='s1_id',how='left').with_columns(pl.col('n').fill_null(0)).select(pl.len().alias('s1'),pl.col('n').sum().alias('candidate_count'),pl.col('n').mean().alias('mean'),pl.col('n').quantile(.95,interpolation='nearest').alias('p95'),pl.col('n').quantile(.99,interpolation='nearest').alias('p99'),pl.col('n').max().alias('max')).row(0,named=True)
        recdict={'candidate_count':dist['candidate_count'],'pair_recall':rec/tcount,'S2_recall':src['S2']['recall'],'S3_recall':src['S3']['recall'],'full_s1_recall':fulls/tper.height,'mean_candidates_per_s1':dist['mean'],'P95':dist['p95'],'P99':dist['p99'],'max':dist['max'],'runtime_seconds':round(time.perf_counter()-tic,2),'peak_memory_gib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3),2),'status':status,'configuration':config,'incremental_recovered_vs_parent':rec,'newly_recovered_vs_parent':matched.join(parent.select('s1_id','candidate_id'),on=['s1_id','candidate_id'],how='anti').height,'newly_missed_vs_parent':truth.join(parent.select('s1_id','candidate_id'),on=['s1_id','candidate_id'],how='anti').join(union.select('s1_id','candidate_id'),on=['s1_id','candidate_id'],how='anti').height}
        return union,recdict
    except Exception as e:
        return parent,{'candidate_count':None,'pair_recall':None,'S2_recall':None,'S3_recall':None,'full_s1_recall':None,'mean_candidates_per_s1':None,'P95':None,'P99':None,'runtime_seconds':round(time.perf_counter()-tic,2),'peak_memory_gib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3),2),'status':'failed','configuration':config,'failure_reason':repr(e)}

manifest=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {'sample_sha256':hashlib.sha256(('\n'.join(IDS)+'\n').encode()).hexdigest(),'sample_size':len(IDS),'experiments':{}}
base_record={'candidate_count':BASE.height,'pair_recall':0.9148381685130015,'S2_recall':0.9179635459916721,'S3_recall':0.9119047218530792,'full_s1_recall':0.7966988727858293,'mean_candidates_per_s1':52.60514,'P95':135,'P99':218,'runtime_seconds':None,'peak_memory_gib':None,'status':'verified_cached','configuration':'existing Phase 2 baseline'}
manifest['experiments']['baseline']=base_record
current=BASE
def run_stage(exp_id, build, cfg):
    global current
    # Resume by loading this experiment's union if available; otherwise build it.
    up=CACHE/(exp_id+'_union.parquet')
    if exp_id in manifest['experiments'] and manifest['experiments'][exp_id].get('status') == 'complete' and up.exists():
        current=pl.read_parquet(up); return
    started=time.perf_counter()
    component=build()
    current,record=compute(exp_id,component,cfg,current)
    record['runtime_seconds']=round(time.perf_counter()-started,2)
    manifest['experiments'][exp_id]=record
    tmp_manifest=MANIFEST.with_suffix('.json.tmp'); tmp_manifest.write_text(json.dumps(manifest,indent=2)+'\n'); tmp_manifest.replace(MANIFEST)
    print(exp_id,json.dumps(record),flush=True)
    if record['status'] != 'complete':
        raise RuntimeError(f"Experiment {exp_id} failed; checkpoint preserved for resume: {record.get('failure_reason')}")

# Address-pair candidates from full-population frequency profiles; profile once and cache.
addr_cache=CACHE/'address_pair_df25k.parquet'
def build_addr_profile():
    all_t=address_token_key_frame(full,'s1_id',2); sample_t=address_token_key_frame(left,'s1_id',2)
    targ_t=address_token_key_frame(targets,'candidate_id',2)
    def capped(keys): return _cap_token_frequency(keys.filter(~pl.col('key').is_in(GENERIC_ADDRESS_TOKENS)),25_000)
    al,asl,ar=map(capped,(all_t,sample_t,targ_t))
    kl,ks,kr=map(token_pair_key_frame,(al,asl,ar))
    shared,_=profile_key_blocks('address_pair_joint_df25k',kl,kr)
    return ks,kr,shared
if addr_cache.exists() and (CACHE/'addr_shared.parquet').exists():
    pass
else:
    ks,kr,shared=build_addr_profile()
    # Cache key-profile output, not source key joins.
    write_parquet_atomic(pl.concat([ks.select(pl.lit('L').alias('side'),pl.col('entity_id').alias('id'),'key'),kr.select(pl.lit('R').alias('side'),pl.col('entity_id').alias('id'),'key')]),CACHE/'addr_keys.parquet',lazy=True)
    write_parquet_atomic(shared,CACHE/'addr_shared.parquet',lazy=True)
if (CACHE/'addr_shared.parquet').exists():
    k=pl.scan_parquet(CACHE/'addr_keys.parquet'); ks=k.filter(pl.col('side')=='L').select(pl.col('id').alias('entity_id'),'key'); kr=k.filter(pl.col('side')=='R').select(pl.col('id').alias('entity_id'),'key')
    shared=pl.scan_parquet(CACHE/'addr_shared.parquet')
    for cap in (1000,2500,5000):
        def b(cap=cap): return _key_pairs(ks,kr,shared,targets,'address_token_pair',cap).collect(engine='streaming')
        run_stage(f'address_pair_cap{cap}',b,{'method':'address_token_pair','component_frequency_ceiling':25000,'joint_pair_cap':cap})

# Name-address joint composite, same capped component universe and measured joint selectivity.
if current is None: raise RuntimeError('no current candidate set')
name_cache=CACHE/'nameaddr_keys.parquet'
if not name_cache.exists() or not (CACHE/'nameaddr_shared.parquet').exists():
    nall=token_key_frame(full,'s1_id','country_normalized',3).filter(~pl.col('key').is_in(LEGAL_FORM_TOKENS)); ns=token_key_frame(left,'s1_id','country_normalized',3).filter(~pl.col('key').is_in(LEGAL_FORM_TOKENS)); nr=token_key_frame(targets,'candidate_id','country_normalized',3).filter(~pl.col('key').is_in(LEGAL_FORM_TOKENS))
    def cap25(k): return _cap_token_frequency(k,25_000)
    nl, nsl, nrr=map(cap25,(nall,ns,nr))
    # Reuse address component keys but derive current sample address keys with identical cap.
    at=address_token_key_frame(left,'s1_id',2).filter(~pl.col('key').is_in(GENERIC_ADDRESS_TOKENS)); ar=address_token_key_frame(targets,'candidate_id',2).filter(~pl.col('key').is_in(GENERIC_ADDRESS_TOKENS))
    atl,atr=map(cap25,(at,ar))
    cl=name_address_composite_key_frame(nsl,atl); cr=name_address_composite_key_frame(nrr,atr)
    # profile uses complete S1 frequencies; make complete composite keys separately.
    af=address_token_key_frame(full,'s1_id',2).filter(~pl.col('key').is_in(GENERIC_ADDRESS_TOKENS)); afl=cap25(af)
    cfl=name_address_composite_key_frame(nl,afl)
    shared,_=profile_key_blocks('name_address_composite_joint_df25k',cfl,cr)
    write_parquet_atomic(pl.concat([cl.select(pl.lit('L').alias('side'),pl.col('entity_id').alias('id'),'key'),cr.select(pl.lit('R').alias('side'),pl.col('entity_id').alias('id'),'key')]),name_cache,lazy=True)
    write_parquet_atomic(shared,CACHE/'nameaddr_shared.parquet',lazy=True)
if (CACHE/'nameaddr_shared.parquet').exists():
    k=pl.scan_parquet(name_cache); kl=k.filter(pl.col('side')=='L').select(pl.col('id').alias('entity_id'),'key'); kr=k.filter(pl.col('side')=='R').select(pl.col('id').alias('entity_id'),'key'); sh=pl.scan_parquet(CACHE/'nameaddr_shared.parquet')
    for cap in (1000,2500):
        run_stage(f'name_address_cap{cap}',lambda cap=cap:_key_pairs(kl,kr,sh,targets,'name_address_composite',cap).collect(engine='streaming'),{'method':'name_address_composite','component_frequency_ceiling':25000,'joint_pair_cap':cap})
        if manifest['experiments'].get(f'name_address_cap{cap}',{}).get('pair_recall',0) >= 0.95:
            print('DEVELOPMENT_TARGET_REACHED; stop tuning this sample', flush=True)
            break

print('CHECKPOINT',MANIFEST,flush=True)
