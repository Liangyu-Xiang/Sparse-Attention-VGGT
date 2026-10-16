#!/usr/bin/env python3
"""Visualize both heads of the highest-ARI adjacent-layer pair, all frames.

Use int32 label rasters (not uint8): 10% clustering has 1036 clusters.
Colours across heads are aligned only for display, by maximum membership
overlap. No labels, values, medoids or ARIs are changed by that permutation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from scripts.visualize_vggt_value_medoids import label, patch_box


def resize_cluster_ids(labels, size):
    return np.asarray(Image.fromarray(np.asarray(labels,dtype=np.int32)).resize(size,Image.Resampling.NEAREST))


def distinct_palette(k,seed=20261015):
    rng = np.random.default_rng(seed)
    colours = np.unique(rng.integers(35,241,(max(2*k,64),3),dtype=np.uint8),axis=0)
    if len(colours)<k:
        raise ValueError("Not enough distinct palette colours")
    return colours[rng.permutation(len(colours))[:k]]


def align_target_colours(source,target,k):
    counts = np.bincount(np.asarray(source,dtype=np.int64)*k+target,minlength=k*k).reshape(k,k)
    source_ids,target_ids = linear_sum_assignment(-counts)
    lookup = np.empty(k,dtype=np.int64)
    lookup[target_ids] = source_ids
    return lookup,counts


def mark_medoids(panel,original,local_medoids,grid):
    draw = ImageDraw.Draw(panel)
    for patch in local_medoids:
        row,col = divmod(int(patch),grid[1])
        box = patch_box(original,row,col,grid)
        x,y = (box[0]+box[2])/2,(box[1]+box[3])/2
        draw.ellipse((x-3,y-3,x+3,y+3),fill="white",outline="black",width=1)


def make_montage(panels,frames,title,subtitle,columns=2):
    width,height = panels[0].size
    rows = (len(panels)+columns-1)//columns
    canvas = Image.new("RGB",(columns*width,100+rows*(height+35)),"white")
    draw = ImageDraw.Draw(canvas)
    label(draw,(12,5),title,23)
    label(draw,(12,43),subtitle,17)
    for index,(panel,frame) in enumerate(zip(panels,frames)):
        x,y = (index%columns)*width,100+(index//columns)*(height+35)
        label(draw,(x+10,y),f"原图帧 {frame}",20)
        canvas.paste(panel,(x,y+30))
    return canvas


def write_csv(path,rows):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=Path("outputs/value_medoids/chess_n10_s5_layers_01_10"))
    args=parser.parse_args()
    metadata=json.loads((args.input_dir/"metadata.json").read_text())
    matrix=np.load(args.input_dir/"all_head_ari_10pct/ari_matrices.npz")["L00_L01"]
    source_head,target_head=map(int,np.unravel_index(matrix.argmax(),matrix.shape))
    heads=[source_head,target_head]
    saved=[np.load(args.input_dir/f"keep_10%/layer_{l:02d}/head_{h:02d}.npz")for l,h in enumerate(heads)]
    medoids=[s["medoids"]for s in saved];assignments=[s["assignments"]for s in saved]
    k=len(medoids[0]);n=len(assignments[0]);frames=metadata["frame_indices"];grid=metadata["patch_grid"]
    per_frame=int(np.prod(grid))
    assert n==len(frames)*per_frame and k==round(.10*n)
    ari=adjusted_rand_score(*assignments)
    np.testing.assert_allclose(ari,matrix[source_head,target_head],atol=1e-12)
    target_lookup,counts=align_target_colours(*assignments,k)
    lookups=[np.arange(k),target_lookup];palette=distinct_palette(k)
    root=args.input_dir/f"best_pair_L00_H{source_head:02d}_L01_H{target_head:02d}_10pct"
    root.mkdir(exist_ok=True)
    np.savez_compressed(root/"display_colour_alignment.npz",counts=counts,target_cluster_to_source_colour=target_lookup,palette_rgb=palette)
    originals=[Image.open(path).convert("RGB")for path in metadata["image_paths"]]
    for original in originals:
        assert grid[1]*14==518 and grid[0]*14==round(original.height*518/original.width/14)*14, "Expected uncropped width-518 inputs"
    preview_indices=[frames.index(frame)for frame in (0,20)]
    patch_rows,cluster_rows,summaries,pdf_pages,head_sections=[],[],[],[],[]
    all_overlays=[]
    for layer,head in enumerate(heads):
        directory=root/f"L{layer:02d}_H{head:02d}"
        directory.mkdir(exist_ok=True)
        labels=assignments[layer];centers=medoids[layer];lookup=lookups[layer]
        sizes=np.bincount(labels,minlength=k)
        np.testing.assert_array_equal(sizes,saved[layer]["cluster_sizes"])
        np.testing.assert_array_equal(labels[centers],np.arange(k))
        assert (sizes>0).all()
        frame_counts=np.stack([np.bincount(lab,minlength=k)for lab in labels.reshape(len(frames),per_frame)])
        np.savez_compressed(directory/"patch_labels.npz",cluster_ids=labels.reshape(len(frames),*grid).astype(np.uint16)+1,
                            medoids=centers,frame_indices=frames,palette_rgb=palette,cluster_to_colour=lookup)
        panels,representatives,pure_panels=[],[],[]
        cards=[]
        title=f"第{layer+1}层 H{head:02d}：{k}个Value簇，全部{n}个patch分类"
        subtitle="颜色=Value簇；白点=medoid；跨头颜色仅按最大交集对齐，不表示簇完全相同"
        for f,(frame,original) in enumerate(zip(frames,originals)):
            raster=resize_cluster_ids(labels.reshape(len(frames),*grid)[f],original.size)
            pure=Image.fromarray(palette[lookup[raster]])
            panel=Image.blend(original,pure,.58)
            representative=original.copy()
            local=centers[centers//per_frame==f]%per_frame
            for image in (panel,representative):mark_medoids(image,original,local,grid)
            panels.append(panel);representatives.append(representative);pure_panels.append(pure)
            for suffix,image in (("clusters",panel),("medoids",representative),("labels",pure)):
                page=make_montage([image],[frame],f"第{layer+1}层 H{head:02d}，帧{frame}；本帧有{len(local)}个medoids",subtitle,columns=1)
                page.save(directory/f"frame_{frame:06d}_{suffix}.png")
                if suffix=="clusters":pdf_pages.append(page)
            cards.append(f'<section><h3>帧{frame}，本帧{len(local)}个代表点</h3><img src="frame_{frame:06d}_clusters.png">'
                         f'<p><a href="frame_{frame:06d}_medoids.png">仅看代表点</a> · <a href="frame_{frame:06d}_labels.png">纯簇标签图</a></p></section>')
        all_overlays.append(panels)
        make_montage(panels,frames,title,subtitle).save(directory/"all_frames_clusters.jpg",quality=95)
        make_montage(representatives,frames,title,"原图上的medoid位置；白点为代表token的patch中心").save(directory/"all_frames_medoids.jpg",quality=95)
        make_montage([panels[i]for i in preview_indices],[frames[i]for i in preview_indices],title,subtitle).save(directory/"preview_clusters.png")
        make_montage([representatives[i]for i in preview_indices],[frames[i]for i in preview_indices],title,"原图上的medoid位置；仅预览第0、20帧，聚类实际使用全部10帧").save(directory/"preview_medoids.png")
        medoid_set=set(centers.tolist())
        for index,cluster in enumerate(labels):
            frame,patch=divmod(index,per_frame);row,col=divmod(patch,grid[1])
            patch_rows.append({"layer":layer,"head":head,"frame":frames[frame],"patch_row":row,"patch_col":col,
                               "cluster_id":int(cluster)+1,"medoid_joint_patch_index":int(centers[cluster]),
                               "distance_to_medoid":float(saved[layer]["distances"][index]),"is_medoid":index in medoid_set})
        for cluster,medoid in enumerate(centers):
            frame,patch=divmod(int(medoid),per_frame);row,col=divmod(patch,grid[1])
            cluster_rows.append({"layer":layer,"head":head,"cluster_id":cluster+1,"patch_count":int(sizes[cluster]),
                                 "medoid_frame":frames[frame],"medoid_patch_row":row,"medoid_patch_col":col,
                                 "frames_containing_members":int((frame_counts[:,cluster]>0).sum()),
                                 **{f"frame_{frame}_members":int(frame_counts[f,cluster])for f,frame in enumerate(frames)}})
        # Explicit per-cluster regions for the four largest clusters, not semantic cherry-picking.
        example_cards=[]
        for cluster in np.argsort(-sizes,kind="stable")[:4]:
            region_panels=[]
            for f,original in enumerate(originals):
                raster=resize_cluster_ids(labels.reshape(len(frames),*grid)[f],original.size)
                mask=Image.fromarray(((raster==cluster)*255).astype(np.uint8))
                dark=Image.fromarray((np.asarray(original)*.12).astype(np.uint8))
                panel=Image.composite(original,dark,mask)
                if centers[cluster]//per_frame==f:mark_medoids(panel,original,[centers[cluster]%per_frame],grid)
                region_panels.append(panel)
            filename=f"cluster_{cluster+1:04d}_members.jpg"
            make_montage(region_panels,frames,f"第{layer+1}层 H{head:02d}，C{cluster+1:04d}：共{sizes[cluster]}个patch",
                         "亮区为该medoid的全部成员；其它区域暗化；按簇大小选择的示例").save(directory/filename,quality=95)
            example_cards.append(f'<p><a href="{filename}">C{cluster+1:04d}的全部原图区域（{sizes[cluster]}个patch）</a></p>')
        summary={"layer":layer,"head":head,"patches":n,"clusters":k,"cluster_size_min":int(sizes.min()),
                 "cluster_size_median":float(np.median(sizes)),"cluster_size_max":int(sizes.max()),
                 "medoids_per_frame":np.bincount(centers//per_frame,minlength=len(frames)).tolist()}
        summaries.append(summary)
        (directory/"index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'+STYLE+
            f'<title>第{layer+1}层H{head:02d}完整聚类</title><h1>{title}</h1><p>{subtitle}</p>'
            '<p>所有10帧联合聚类；不是语义分割，也不是只聚类10%的token。1036种颜色不保证肉眼都能区分；精确归属见NPZ/CSV。</p>'
            '<p><a href="../index.html">返回头对比较</a> · <a href="all_frames_clusters.jpg">全部10帧分类总图</a> · '
            '<a href="all_frames_medoids.jpg">全部10帧代表点位置</a> · <a href="patch_labels.npz">完整标签数组</a></p>'
            '<h2>四个最大簇的成员区域</h2>'+''.join(example_cards)+'<h2>逐帧结果</h2><div class="grid">'+''.join(cards)+'</div></html>')
        head_sections.append(f'<h2>第{layer+1}层 H{head:02d}</h2><p><a href="L{layer:02d}_H{head:02d}/index.html">查看全部10帧和单簇区域</a></p>'
                             f'<img src="L{layer:02d}_H{head:02d}/preview_clusters.png">')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    write_csv(root/"all_patch_classes.csv",patch_rows);write_csv(root/"cluster_regions.csv",cluster_rows)
    pdf_pages[0].save(root/"both_heads_all_frames.pdf",save_all=True,append_images=pdf_pages[1:],resolution=110)
    summary={"metadata":metadata,"source_layer":0,"source_head":source_head,"target_layer":1,"target_head":target_head,
             "selection":"maximum ARI over all 16x16 layer-1/layer-2 head pairs","ari":float(ari),"retention":.1,
             "heads":summaries,"display_alignment":"Hungarian maximum membership overlap, display colours only; no assumed equal cluster IDs",
             "label_storage":"uint16 1-based IDs in NPZ; int32 nearest-neighbour rasters, no uint8 truncation",
             "preview_frames":[0,20],"all_frames":frames,"same_label_after_display_alignment":float(counts[np.arange(k),np.argsort(target_lookup)].sum()/n)}
    (root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    (root/"index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'+STYLE+
        f'<title>第1-2层最高ARI头对的聚类结果</title><h1>第1层H{source_head:02d}与第2层H{target_head:02d}：ARI={ari:.6f}</h1>'
        f'<p>在全部16×16头对中最高。每头10360个patch、1036个medoids（10%）；每个头独立聚类。白点是代表点，颜色是Value簇。'
        '颜色按全10帧的最大簇交集对齐，仅方便阅读，不表示前后簇编号或成员完全相同。下图先预览第0、20帧。</p>'
        '<p><a href="both_heads_all_frames.pdf">全部20页分类PDF</a> · <a href="all_patch_classes.csv">20720条patch分类</a> · '
        '<a href="cluster_regions.csv">2072个簇的统计</a></p>'+''.join(head_sections)+'</html>')
    print(f"Output: {root}",flush=True)


STYLE='<style>body{max-width:1500px;margin:25px auto;padding:12px;font-family:system-ui;line-height:1.7}img{max-width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}a{color:#14609a}</style>'


if __name__=="__main__":
    main()
