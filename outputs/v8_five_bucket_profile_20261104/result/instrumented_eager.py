def eager(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,cap,iterations,drop_multiplier):
    _profile_mark(0)
    context=rule.gate(prior,pi,population,old_ids,labels,reps)
    _profile_mark(1)
    if geometry.get('fuse_state_kernel',False) and geometry['width']<=64:
        from saf3r.triton.reuse_fused_states import states
        errors,costs=states(prior,z,reps,labels,population,allowed,geometry,context)
    else:
        geometry['drop_values']=drop_norm(prior,z,geometry,len(reps),geometry['drop_bins'])
        errors,costs=rule.states(pi,z,reps,labels,population,None,allowed,prior,drop_multiplier,geometry,context)
    _profile_mark(2)
    selected,count=budget(geometry['objective'],costs,errors[:,:,0].amax(1)+1e-12,cap,iterations)
    selected=rule.decode(selected)
    _profile_mark(3)
    ids,real,record=rule.pack(selected,None,reps,labels,population,cap,context)
    _profile_mark(4)
    weights=rule.transfer(selected,labels,reps,None,prior,pi,ids,geometry,context)
    _profile_mark(5)
    record.update(rule.diagnostics(errors,selected,None,old_ids,labels,reps,context))
    record['history_cap_discarded']=torch.zeros(len(prior),device=prior.device)
    _profile_mark(6)
    return ids,real,weights,record
