"""Profile-only scopes and graph-safe device timestamp instrumentation.

Production functions remain on disk unchanged. AST scopes preserve original statements.
Graph markers write a separate per-replay row, so timings are not overwritten by reuse.
"""
import ast
from contextlib import contextmanager
from functools import wraps
import importlib
import inspect
from pathlib import Path
import textwrap

import torch
import triton
import triton.language as tl

GRAPH_NAMES=('gate','history_loss_corrections','four_state_errors','budget16','decode','pack','diagnostics',
             'pack_flags','pack_prefix','pack_validate','pack_emit','pack_output_statistics')
COLS=2+2*len(GRAPH_NAMES)

@triton.jit
def _mark(TIMES,META,COUNTER,LAYER,SLOT:tl.constexpr,COLS:tl.constexpr,LIMIT:tl.constexpr,
          ROOT:tl.constexpr=False,END:tl.constexpr=False,CAPTURED:tl.constexpr=False,GROUPS:tl.constexpr=0):
    row=tl.load(COUNTER)
    tick=tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;',constraints='=l',args=[],dtype=tl.uint64,is_pure=False,pack=1)
    tl.store(TIMES+row*COLS+SLOT,tick,row<LIMIT)
    if ROOT:
        layer=tl.load(LAYER)
        tl.store(META+row*3,tl.full((),1 if CAPTURED else 0,tl.int32),row<LIMIT)
        tl.store(META+row*3+1,layer,row<LIMIT)
        tl.store(META+row*3+2,tl.full((),GROUPS,tl.int32),row<LIMIT)
    if END:tl.store(COUNTER,row+1)


class GraphRecorder:
    def __init__(self,capacity=16384):
        self.capacity=capacity
        self.times=torch.zeros((capacity,COLS),device='cuda',dtype=torch.int64)
        self.meta=torch.zeros((capacity,3),device='cuda',dtype=torch.int32)
        self.counter=torch.zeros((),device='cuda',dtype=torch.int32)
        self.layer=torch.zeros((),device='cuda',dtype=torch.int32)
        self.seconds_per_tick=1e-9
        self.calibration=self.calibrate()
    def marker(self,slot,**kwargs):
        _mark[(1,)](self.times,self.meta,self.counter,self.layer,slot,COLS,self.capacity,num_warps=1,**kwargs)
    @contextmanager
    def region(self,name):
        slot=2+GRAPH_NAMES.index(name)*2
        self.marker(slot)
        try:yield
        finally:self.marker(slot+1)
    def reset(self):
        self.counter.zero_()
    def calibrate(self):
        # The CUDA Event and device-clock intervals contain the same GEMMs.
        # Warm the marker and GEMM first, then check ns ticks against CUDA Events.
        x=torch.randn(2048,2048,device='cuda');y=torch.randn_like(x)
        self.marker(0,ROOT=True);z=x@y;self.marker(1,END=True)
        torch.cuda.synchronize();self.reset()
        rows=[]
        for _ in range(3):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();self.marker(0,ROOT=True)
            for j in range(8):z=x@y
            self.marker(1,END=True);end.record();torch.cuda.synchronize()
            ticks=self.times[int(self.counter.item())-1,:2].cpu().tolist()
            seconds=(ticks[1]-ticks[0])*1e-9
            event=begin.elapsed_time(end)/1000
            assert seconds>0 and abs(seconds-event)<max(.00005,event*.03),(seconds,event)
            rows.append(dict(device_tick_seconds=seconds,cuda_event_seconds=event))
        self.reset();del x,y,z
        return rows
    def collect(self):
        n=int(self.counter.item());assert n<=self.capacity
        stamps=self.times[:n].cpu().tolist();meta=self.meta[:n].cpu().tolist()
        captured=[];warm=[]
        for ticks,(is_captured,layer,groups) in zip(stamps,meta):
            if not ticks[0] or ticks[1]<ticks[0]:raise AssertionError('Incomplete graph record')
            row=dict(layer=layer,groups=groups,total_seconds=(ticks[1]-ticks[0])*self.seconds_per_tick,stages={})
            for i,name in enumerate(GRAPH_NAMES):
                start,stop=ticks[2+i*2:4+i*2]
                if start:
                    assert stop>=start,name
                    row['stages'][name]=(stop-start)*self.seconds_per_tick
            (captured if is_captured else warm).append(row)
        totals={name:sum(r['stages'].get(name,0.) for r in captured) for name in GRAPH_NAMES}
        return dict(captured_rows=len(captured),eager_warm_rows=len(warm),
            total_seconds=sum(r['total_seconds'] for r in captured),stages_seconds=totals,
            rows=captured,eager_warm_rows_detail=warm,
            seconds_per_tick=self.seconds_per_tick,calibration=self.calibration,
            note='Graph-safe device timestamps per actual replay, not last-replay estimates. Marker launch gaps included. pack subrows expand pack; never add parent and child.')


def scoped_ast(owner,name,scope,transform):
    original=getattr(owner,name)
    src=textwrap.dedent(inspect.getsource(original));tree=ast.parse(src)
    globals_dict=original.__globals__
    globals_dict['_deep_scope']=scope
    fn=tree.body[0]
    transform(fn,src)
    ast.fix_missing_locations(tree)
    namespace={}
    exec(compile(tree,inspect.getsourcefile(original),'exec'),globals_dict,namespace)
    replacement=namespace[name]
    replacement.__doc__=original.__doc__
    setattr(owner,name,replacement)
    return original


def block(label,statements):
    node=ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='_deep_scope',ctx=ast.Load()),
        args=[ast.Constant(label)],keywords=[]),optional_vars=None)],body=statements,type_comment=None)
    return ast.copy_location(node,statements[0])


def patch_function_scope(owner,name,label,scope):
    original=getattr(owner,name)
    @wraps(original)
    def replacement(*args,**kwargs):
        with scope(label):return original(*args,**kwargs)
    setattr(owner,name,replacement)
    return original


def install_clustering(scope):
    ac=importlib.import_module('saf3r.patch.adaptive_value_clustering')
    def split_loop(marker,first,second):
        def transform(fn,src):
            lines=src.splitlines();cut=next(i+1 for i,x in enumerate(lines) if marker in x)
            loop=next(n for n in fn.body if isinstance(n,ast.For))
            before=[n for n in loop.body if n.lineno<cut]
            after=[n for n in loop.body if n.lineno>=cut]
            assert before and after
            loop.body=[block(first,before),block(second,after)]
        return transform
    scoped_ast(ac,'center_graph',scope,split_loop('v,ix=dist.topk','cluster_sub/center_graph_distances','cluster_sub/center_graph_topk'))
    scoped_ast(ac,'full_assignment',scope,split_loop('best,idx=dist.min','cluster_sub/global_distances','cluster_sub/global_argmin_assignment'))
    scoped_ast(ac,'local_assignment',scope,split_loop('radius=own.sqrt','cluster_sub/local_gather_distance_assignment','cluster_sub/local_certificate_math'))
    def lloyd_transform(fn,src):
        lines=src.splitlines()
        line=lambda token:next(i+1 for i,x in enumerate(lines) if token in x)
        loop=next(n for n in fn.body if isinstance(n,ast.For))
        new=[]
        for stmt in loop.body:
            if stmt.lineno==line("if bool((population==0).any())"):
                new.append(block('cluster/nonempty_check_sync',[stmt]))
            else:new.append(stmt)
        loop.body=new
        # Inside the local branch, count certificate conversion/nonzero separately.
        branch=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare))
        start=line("if mode=='local' and iteration")
        stop=line('if fallback:')
        i=next(j for j,n in enumerate(branch.orelse) if n.lineno==start)
        j=next(j for j,n in enumerate(branch.orelse) if n.lineno==stop)
        branch.orelse=branch.orelse[:i]+[block('cluster/local_diagnostics_sync',branch.orelse[i:j])]+branch.orelse[j:]
        start=line('changed=int(');stop=line('records.append(')
        i=next(j for j,n in enumerate(loop.body) if n.lineno==start)
        j=next(j for j,n in enumerate(loop.body) if n.lineno==stop)
        loop.body=loop.body[:i]+[block('cluster/change_count_sync',loop.body[i:j])]+loop.body[j:]
        start=line('distance=(x-centers[labels])')
        i=next(j for j,n in enumerate(fn.body) if n.lineno==start)
        j=next(j for j,n in enumerate(fn.body) if isinstance(n,ast.Return))
        fn.body=fn.body[:i]+[block('cluster/final_real_representatives',fn.body[i:j])]+fn.body[j:]
    scoped_ast(ac,'lloyd',scope,lloyd_transform)
    for name,label in (
        ('centers_from_labels','cluster/means_population'),('center_graph','cluster/center_neighbor_graph'),
        ('full_assignment','cluster/L0_global_assignment'),('local_assignment','cluster/local_assignment'),
        ('repair_empty','cluster/empty_repair')):
        patch_function_scope(ac,name,label,scope)


def install_prior(scope):
    history=importlib.import_module('saf3r.patch.reuse_sampled_history')
    def transform(fn,src):
        lines=src.splitlines()
        cuts=[('current=obs[','prior_detail/grid_beta_and_uniform'),
              ('prior=base[:,s.grid]','prior_detail/expand_all_patch_prior'),
              ('predict(s.ready_edges','prior_detail/history_predict'),
              ('pi=torch.zeros','prior_detail/cluster_scatter'),
              ('s.m5c_prior_errors.append','prior_detail/normalization_check')]
        positions=[next(i+1 for i,s in enumerate(lines) if token in s) for token,_ in cuts]
        prefixes=[n for n in fn.body if n.lineno<positions[0]]
        returns=[n for n in fn.body if isinstance(n,ast.Return)]
        body=prefixes
        for i,(_,label) in enumerate(cuts):
            stop=positions[i+1] if i+1<len(positions) else returns[0].lineno
            stmts=[n for n in fn.body if positions[i]<=n.lineno<stop]
            assert stmts
            body.append(block(label,stmts))
        fn.body=body+returns
    scoped_ast(history,'build_prior',scope,transform)
    patch_function_scope(history,'build_prior','selection/prior_total',scope)


def install_graph_detail(recorder):
    selector=importlib.import_module('saf3r.patch.reuse_window_graph')
    rule=importlib.import_module('saf3r.patch.reuse_inference_ops')
    shared=importlib.import_module('saf3r.triton.reuse_shared_value_states')
    original_eager=selector.eager
    # Capture a deterministic small reference BEFORE replacing production aliases.
    from scripts.validate_shared_value_loss import example
    z,labels,grid,reps,pop,uniform,prior,old_ids,pi,*_=example('cuda')
    from saf3r.triton.reuse_drop_moment import prepare
    geom=shared.prepare_geometry(z,labels,reps,pop);geom['drop_bins']=prepare(pop)
    geom['fuse_state_kernel']=True;shared.prepare(z,labels,reps,grid,uniform.shape[1],geom)
    allowed=torch.ones_like(pi,dtype=torch.bool)
    expected=original_eager(prior,pi,old_ids,allowed,z,reps,labels,pop,geom,24,16,1.,uniform=uniform)
    def pack_transform(fn,src):
        lines=src.splitlines()
        cuts=[('kernels._flags[','pack_flags'),('prefix=flags.cumsum','pack_prefix'),
              ('kernels._validate[','pack_validate'),('_emit[','pack_emit'),('return ids,','pack_output_statistics')]
        positions=[next(i+1 for i,s in enumerate(lines) if token in s) for token,_ in cuts]
        body=[n for n in fn.body if n.lineno<positions[0]]
        for i,(_,label) in enumerate(cuts):
            stop=positions[i+1] if i+1<len(positions) else 10**9
            stmts=[n for n in fn.body if positions[i]<=n.lineno<stop]
            assert stmts
            body.append(block(label,stmts))
        fn.body=body
    scoped_ast(rule,'pack',recorder.region,pack_transform)
    for owner,name,label in ((rule,'gate','gate'),(shared,'corrections','history_loss_corrections'),
            (shared,'states','four_state_errors'),(selector,'budget','budget16'),
            (rule,'decode','decode'),(rule,'pack','pack'),(rule,'diagnostics','diagnostics')):
        patch_function_scope(owner,name,label,recorder.region)
    @wraps(original_eager)
    def eager(*args,**kwargs):
        captured=torch.cuda.is_current_stream_capturing()
        if not captured:
            # Load the capture specialization BEFORE entering CUDA capture.
            # The next mark overwrites this warmup metadata in the same row.
            recorder.marker(0,ROOT=True,CAPTURED=True,GROUPS=len(args[0]))
        recorder.marker(0,ROOT=True,CAPTURED=captured,GROUPS=len(args[0]))
        try:return original_eager(*args,**kwargs)
        finally:recorder.marker(1,END=True)
    selector.eager=eager
    # Verify real CUDA Graph outputs/supports, not just mathematical proxy values.
    window=selector.Window(prior,pi,old_ids,allowed,z,reps,labels,pop,geom,24,16,1.,uniform=uniform)
    for _ in range(3):
        actual=window(prior,pi,old_ids,allowed,uniform=uniform)
        for a,b in zip(actual[:3],expected[:3]):torch.testing.assert_close(a,b,atol=0,rtol=0)
        for key in expected[3]:torch.testing.assert_close(actual[3][key],expected[3][key],atol=1e-4,rtol=1e-4)
    torch.cuda.synchronize();checked=recorder.collect()
    assert checked['captured_rows']==3,checked['captured_rows']
    recorder.reset()
    return dict(passed=True,captured_replays_checked=3,exact_supports_and_weights=True,record_statistics_close=True)


def install_window_host_scopes(scope):
    selector=importlib.import_module('saf3r.patch.reuse_window_graph')
    def call_transform(fn,src):
        lines=src.splitlines()
        line=lambda token:next(i+1 for i,x in enumerate(lines) if token in x)
        copy_line=line('for dest,src in zip(');replay_line=line('self.graph.replay()');return_line=line('return self.ids')
        body=[]
        for n in fn.body:
            if n.lineno==copy_line:body.append(block('graph_host/input_copies',[n]))
            elif n.lineno==replay_line:body.append(block('graph_host/replay',[n]))
            elif replay_line<n.lineno<return_line:
                if not body or not (isinstance(body[-1],ast.With) and isinstance(body[-1].items[0].context_expr,ast.Call) and body[-1].items[0].context_expr.args[0].value=='graph_host/record_copies'):
                    body.append(block('graph_host/record_copies',[n]))
                else:body[-1].body.append(n)
            else:body.append(n)
        fn.body=body
    scoped_ast(selector.Window,'__call__',scope,call_transform)
    creating=dict(active=False)
    old_init=selector.Window.__init__
    @wraps(old_init)
    def init(self,*args,**kwargs):
        creating['active']=True
        try:
            with scope('graph_host/create_capture'):return old_init(self,*args,**kwargs)
        finally:creating['active']=False
    selector.Window.__init__=init
    old_update=selector.Window.update_constants
    @wraps(old_update)
    def update(self,*args,**kwargs):
        name='graph_host/geometry_copies_during_create' if creating['active'] else 'graph_host/geometry_copies'
        with scope(name):return old_update(self,*args,**kwargs)
    selector.Window.update_constants=update
    patch_function_scope(selector,'select','selection/selector_total',scope)


def install_geometry(scope):
    shared=importlib.import_module('saf3r.triton.reuse_shared_value_states')
    drop=importlib.import_module('saf3r.triton.reuse_drop_moment')
    core=importlib.import_module('saf3r.patch.reuse_residual_exploration')
    patch_function_scope(shared,'prepare_geometry','geometry/value_errors',scope)
    patch_function_scope(shared,'prepare','geometry/cluster_grid_statistics',scope)
    patch_function_scope(drop,'prepare','geometry/population_bins',scope)
    patch_function_scope(core.ReuseExploreState,'prepare','geometry/population_grid_pool_overlap',scope)


def install_deep(model,state,scope):
    install_clustering(scope);install_prior(scope);install_geometry(scope);install_coarse_and_candidates(scope)
    recorder=GraphRecorder()
    with torch.inference_mode():
        validation=install_graph_detail(recorder)
    install_window_host_scopes(scope)
    handles=[]
    for idx,b in enumerate(model.aggregator.global_blocks):
        def tag(module,args,idx=idx):recorder.layer.fill_(idx)
        handles.append(b.attn.register_forward_pre_hook(tag))
    # Existing warm graphs have no markers. Recapture only for the separate audit.
    if hasattr(state,'_reused_window_graphs'):state._reused_window_graphs.clear()
    if hasattr(state,'_window_graphs'):state._window_graphs.clear()
    return recorder,validation,handles


def install_coarse_and_candidates(scope):
    cache=importlib.import_module('saf3r.patch.reuse_probe_cache')
    direct=importlib.import_module('saf3r.patch.reuse_direct_coarse_observation')
    core=importlib.import_module('saf3r.patch.reuse_residual_exploration')
    def coarse_transform(fn,src):
        lines=src.splitlines()
        cut1=next(i+1 for i,x in enumerate(lines) if 'grid=torch.einsum' in x)
        cut2=next(i+1 for i,x in enumerate(lines) if "if getattr(getattr(s,'config'" in x)
        ret=next(n for n in fn.body if isinstance(n,ast.Return))
        segments=[('coarse/probe_gather',[n for n in fn.body if n.lineno<cut1]),
            ('coarse/grid_special_qk',[n for n in fn.body if cut1<=n.lineno<cut2]),
            ('coarse/normalization',[n for n in fn.body if cut2<=n.lineno<ret.lineno])]
        fn.body=[block(label,stmts) for label,stmts in segments]+[ret]
    scoped_ast(cache,'observe',scope,coarse_transform)
    direct._observe=cache.observe
    def direct_transform(fn,src):
        fn.body=[block('coarse/patch_only_probability',[n]) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript) else n for n in fn.body]
    scoped_ast(direct,'observe',scope,direct_transform)
    def core_transform(fn,src):
        # Add scopes only inside existing coarse/fine/value parent scopes.
        for n in ast.walk(fn):
            if not isinstance(n,ast.With):continue
            call=n.items[0].context_expr
            if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='stage':continue
            if not isinstance(call.args[1],ast.Constant):continue
            tag=call.args[1].value
            if tag=='value_update':
                n.body=[block('cluster/head_mean_value_features',[stmt]) if isinstance(stmt,ast.Assign) else stmt for stmt in n.body]
            if tag=='residual_fine_probe':
                labels=('candidate/grid_to_cluster_mm','candidate/topk_clusters','candidate/allow_flags')
                assert all(isinstance(stmt,ast.Assign) for stmt in n.body[:3])
                n.body=[block(label,[stmt]) for label,stmt in zip(labels,n.body[:3])]+n.body[3:]
    scoped_ast(core.ReuseResidualExploration,'forward',scope,core_transform)
