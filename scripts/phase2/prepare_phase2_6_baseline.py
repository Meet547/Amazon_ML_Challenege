"""Prepare the fixed-sample Phase 2.6 baseline and initialize its checkpoint manifest."""
import hashlib, json, resource, time, shutil, gc
from pathlib import Path
import polars as pl

from src.phase2.loader import load_entity_tables
from src.phase2.blocks import (
    GENERIC_ADDRESS_TOKENS, LEGAL_FORM_TOKENS, source2_and_source3,
    exact_key_frame, token_key_frame, name_token_pair_key_frame,
    address_token_key_frame, numeric_address_key_frame,
    core_name_key_frame, sorted_name_token_key_frame,
    token_pair_key_frame, name_address_composite_key_frame, profile_key_blocks,
)
from src.phase2.candidate_generator import (
    _source1, _name_pairs, _address_pairs, _rare_token_pairs, _key_pairs,
    _cap_token_frequency,
)
from src.phase2.deduplicate import deduplicate_with_provenance
from src.phase2.evaluate import evaluate_recall
from src.phase2.config import NAME_TOKEN_MIN_LENGTH, NAME_TOKEN_MAX_PAIR_ESTIMATE
from src.evaluation.ground_truth import load_ground_truth

ROOT=Path.cwd()
import os
TMP=Path(os.environ.get('PHASE2_6_CACHE_DIR', '/private/tmp/phase2_6_cache')); TMP.mkdir(parents=True, exist_ok=True)
if (TMP/'baseline.parquet').exists():
    print('Baseline cache already exists:', TMP/'baseline.parquet', flush=True)
    raise SystemExit(0)
METRICS=ROOT/'outputs/metrics/phase2_6_experiments.json'
CAP=1_000; BASELINE_COMPONENT_CAP=10_000; t0=time.perf_counter(); disk0=shutil.disk_usage(ROOT).free
sample_ids=(ROOT/'outputs/metrics/phase2_100k_sample_ids.txt').read_text().splitlines()
if len(sample_ids)!=100_000: raise RuntimeError('Expected frozen 100K development IDs')
sample_sha=hashlib.sha256(('\n'.join(sample_ids)+'\n').encode()).hexdigest()
tables=load_entity_tables('train'); s1full=_source1(tables)
s1= s1full.filter(pl.col('s1_id').is_in(sample_ids))
targets=source2_and_source3(tables).with_columns(pl.col('candidate_id').str.slice(0,2).alias('candidate_source'))
labels=load_ground_truth(ROOT/'datasets/train/train_ground_truth.tsv',ROOT/'datasets/train/train_source1.tsv',ROOT/'datasets/train/train_source2.tsv',ROOT/'datasets/train/train_source3.tsv').filter(pl.col('source1_entity_id').is_in(sample_ids))
truth=labels.lazy().select(pl.col('source1_entity_id').alias('s1_id'),pl.col('matched_entity_ids').alias('candidate_id')).explode('candidate_id',empty_as_null=True).filter(pl.col('candidate_id').is_not_null()).collect(engine='streaming')
sample_frame=pl.DataFrame({'s1_id':sample_ids})

def measure(pairs: pl.DataFrame):
    rec=evaluate_recall(pairs.lazy(),labels)
    counts=pairs.group_by('s1_id').len().rename({'len':'n'})
    v=sample_frame.join(counts,on='s1_id',how='left').with_columns(pl.col('n').fill_null(0))
    vol=v.select(pl.col('n').sum().alias('total'),pl.col('n').mean().alias('mean'),pl.col('n').median().alias('median'),pl.col('n').quantile(.9,interpolation='nearest').alias('p90'),pl.col('n').quantile(.95,interpolation='nearest').alias('p95'),pl.col('n').quantile(.99,interpolation='nearest').alias('p99'),pl.col('n').max().alias('max'),(pl.col('n')==0).sum().alias('zero')).row(0,named=True)
    return {'volume':vol,'recall':rec}

def union(base: pl.DataFrame, extras):
    frames=[base.lazy()]+[x.lazy() for x in extras if x is not None]
    return deduplicate_with_provenance(pl.concat(frames,how='vertical')).collect(engine='streaming')

def add_measure(name, base, candidates, prior):
    combined=union(base,[candidates])
    mm=measure(combined)
    new=combined.select('s1_id','candidate_id').join(base.select('s1_id','candidate_id'),on=['s1_id','candidate_id'],how='anti')
    delta=truth.join(new,on=['s1_id','candidate_id'],how='inner').height
    report={'configuration':name,'metrics':mm,'incremental_true_pairs_vs_prior':delta,'added_unique_candidates_vs_prior':new.height}
    print(name,json.dumps(report,default=str),flush=True)
    return combined,report

def profile(method,left,right,cap=CAP):
    shared,stats=profile_key_blocks(method,left,right,pair_cap=cap)
    return shared,stats

results=[]; raw={}; stats={}
print('Building frozen baseline profiles',flush=True)
nl=exact_key_frame(s1full,'business_name_normalized','s1_id','key'); nr=exact_key_frame(targets,'business_name_normalized','candidate_id','key')
_,stats['exact_name']=profile_key_blocks('exact_name',nl,nr,None)
al=exact_key_frame(s1full,'business_address_normalized','s1_id','key'); ar=exact_key_frame(targets,'business_address_normalized','candidate_id','key')
_,stats['exact_address']=profile_key_blocks('exact_address',al,ar,None)
name_l=token_key_frame(s1full,'s1_id','country_normalized',NAME_TOKEN_MIN_LENGTH)
name_s=token_key_frame(s1,'s1_id','country_normalized',NAME_TOKEN_MIN_LENGTH)
name_r=token_key_frame(targets,'candidate_id','country_normalized',NAME_TOKEN_MIN_LENGTH).join(targets.select(pl.col('candidate_id').alias('entity_id'),'candidate_source'),on='entity_id',how='inner')
shared_name,stats['rare_name_token']=profile('rare_name_token',name_l,name_r,NAME_TOKEN_MAX_PAIR_ESTIMATE)
np_l=name_token_pair_key_frame(s1full,'s1_id',3); np_s=name_token_pair_key_frame(s1,'s1_id',3); np_r=name_token_pair_key_frame(targets,'candidate_id',3)
shared_np,stats['rare_name_token_pair']=profile('rare_name_token_pair',np_l,np_r)
at_l=address_token_key_frame(s1full,'s1_id',2); at_s=address_token_key_frame(s1,'s1_id',2); at_r=address_token_key_frame(targets,'candidate_id',2)
shared_at,stats['address_token']=profile('address_token',at_l,at_r)
num_l=numeric_address_key_frame(s1full,'s1_id',2); num_s=numeric_address_key_frame(s1,'s1_id',2); num_r=numeric_address_key_frame(targets,'candidate_id',2)
shared_num,stats['numeric_address']=profile('numeric_address',num_l,num_r)
core_l=core_name_key_frame(s1full,'s1_id'); core_s=core_name_key_frame(s1,'s1_id'); core_r=core_name_key_frame(targets,'candidate_id')
shared_core,stats['name_core_exact']=profile('name_core_exact',core_l,core_r)
sort_l=sorted_name_token_key_frame(s1full,'s1_id'); sort_s=sorted_name_token_key_frame(s1,'s1_id'); sort_r=sorted_name_token_key_frame(targets,'candidate_id')
shared_sort,stats['name_sorted_tokens']=profile('name_sorted_tokens',sort_l,sort_r)
addr_info_l=_cap_token_frequency(at_l.filter(~pl.col('key').is_in(GENERIC_ADDRESS_TOKENS)),BASELINE_COMPONENT_CAP)
addr_info_s=_cap_token_frequency(at_s.filter(~pl.col('key').is_in(GENERIC_ADDRESS_TOKENS)),BASELINE_COMPONENT_CAP)
addr_info_r=_cap_token_frequency(at_r.filter(~pl.col('key').is_in(GENERIC_ADDRESS_TOKENS)),BASELINE_COMPONENT_CAP)
ap_l=token_pair_key_frame(addr_info_l); ap_s=token_pair_key_frame(addr_info_s); ap_r=token_pair_key_frame(addr_info_r)
shared_ap,stats['address_token_pair']=profile('address_token_pair',ap_l,ap_r)
name_info_l=_cap_token_frequency(token_key_frame(s1full,'s1_id','country_normalized',3).filter(~pl.col('key').is_in(LEGAL_FORM_TOKENS)),BASELINE_COMPONENT_CAP)
name_info_s=_cap_token_frequency(token_key_frame(s1,'s1_id','country_normalized',3).filter(~pl.col('key').is_in(LEGAL_FORM_TOKENS)),BASELINE_COMPONENT_CAP)
name_info_r=_cap_token_frequency(token_key_frame(targets,'candidate_id','country_normalized',3).filter(~pl.col('key').is_in(LEGAL_FORM_TOKENS)),BASELINE_COMPONENT_CAP)
comp_l=name_address_composite_key_frame(name_info_l,addr_info_l); comp_s=name_address_composite_key_frame(name_info_s,addr_info_s); comp_r=name_address_composite_key_frame(name_info_r,addr_info_r)
shared_comp,stats['name_address_composite']=profile('name_address_composite',comp_l,comp_r)

raw['exact_name']=_name_pairs(s1,targets)
raw['exact_address']=_address_pairs(s1,targets)
raw['rare_name_token']=_rare_token_pairs(s1,targets,name_s,name_r,shared_name,NAME_TOKEN_MAX_PAIR_ESTIMATE)
for method,l,r,sh in [('rare_name_token_pair',np_s,np_r,shared_np),('address_token',at_s,at_r,shared_at),('address_token_pair',ap_s,ap_r,shared_ap),('name_address_composite',comp_s,comp_r,shared_comp),('numeric_address',num_s,num_r,shared_num),('name_core_exact',core_s,core_r,shared_core),('name_sorted_tokens',sort_s,sort_r,shared_sort)]:
    raw[method]=_key_pairs(l,r,sh,targets,method,CAP)
baseline=deduplicate_with_provenance(pl.concat(list(raw.values()),how='vertical')).collect(engine='streaming')
baseline.write_parquet(TMP/'baseline.parquet',compression='zstd')
baseline_metrics=measure(baseline)
print('BASELINE',json.dumps(baseline_metrics,default=str),flush=True)
report={'sample_sha256':sample_sha,'sample_size':len(sample_ids),'experiments':{'baseline':{'candidate_count':baseline_metrics['volume']['total'],'pair_recall':baseline_metrics['recall']['candidate_recall'],'S2_recall':baseline_metrics['recall']['by_candidate_source']['S2']['recall'],'S3_recall':baseline_metrics['recall']['by_candidate_source']['S3']['recall'],'full_s1_recall':baseline_metrics['recall']['positive_s1_recovery']['fully_recovered_s1_pct'],'mean_candidates_per_s1':baseline_metrics['volume']['mean'],'P95':baseline_metrics['volume']['p95'],'P99':baseline_metrics['volume']['p99'],'max':baseline_metrics['volume']['max'],'runtime_seconds':round(time.perf_counter()-t0,2),'peak_memory_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3),'status':'complete','configuration':'Phase 2 baseline; address/name composite component cap 10000, joint cap 1000'}},'profile_stats':{k:{x:y for x,y in v.items() if x!='largest_shared_blocks'} for k,v in stats.items()}}
tmp_manifest=METRICS.with_suffix('.json.tmp');tmp_manifest.write_text(json.dumps(report,indent=2,default=str)+'\n');tmp_manifest.replace(METRICS)
print('BASELINE_CACHE_READY', TMP/'baseline.parquet', flush=True)
