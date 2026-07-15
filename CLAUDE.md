# CLAUDE.md

CAGE / RoomFormer 点云 → 2D 户型图重建。本文件记录本仓库自定义推理链路的关键约定，供后续开发/调试参考。

## 开发环境

- 本地已配置 conda 环境 **`CAGE`**：开发与调试前先 `conda activate CAGE`。已装 `numpy opencv-python matplotlib plyfile shapely`（可跑 `align_floorplan.py` 全流程）。
- **注意**：`torch` 与模型权重 `checkpoint/CAGE_stru3d_swinv2.pth` 尚未安装，故 `infer_pointcloud.py` 的模型推理暂不能在本机实跑；`align_floorplan.py` 不依赖 torch，可实跑。
- numpy 版本为 1.24，已移除 `np.int` 等旧别名——新增代码勿用。
- 说明文档统一放在 **`docs/`** 文件夹（见 `docs/align_floorplan.md`）。

## 自定义点云推理链路

入口：`infer_pointcloud.py`（包装脚本 `tools/infer_pointcloud.sh`）。
把任意 `.ply` 点云投影成 256×256 密度图 → RoomFormer 模型 → 房间多边形，输出
`infer_out/{name}_density.png`、`_pred.png`、`_polys.json`。

### 正向变换链（`load_input`）

原始 `.ply` `xyz`
→ `reorder_up_axis(up_axis)`：列置换，把竖直（up）轴移到第 3 列（`y` 轴时 `xyz[:,[0,2,1]]`）
→ 离群点裁剪（见下）
→ `estimate_yaw` + `rotate_floor_plane(+yaw)`：只旋转第 0/1 列，把楼面转正
→ `generate_density`：投影 `xyz[:,:2]` 成密度图（`stru3d_utils.py`，末尾 `density /= density.max()` 归一化到 [0,1]）
→ **`hflip`（左右翻转）**：见下第 4 点。y-up 的列置换 `[0,2,1]` 是奇置换、翻了手性，
  投影成"从下往上"看的镜像；`density_from_xyz(hflip=True)` 对密度图做 `np.fliplr` 转成常规俯视。

`_polys.json` 里的 `world_mm/world_m` = **重排 + yaw 校正后**的 2D 楼面坐标，不是原始点云坐标。
（`hflip` 只翻图像，`pixel_to_world` 用 `255-col` 反算把它抵消，故 `world_mm` 与不翻时**逐位相同**，`polys_to_3d` 不受影响。）

## 本仓库已做的关键修复 / 增强（infer_pointcloud.py）

针对 MVS / Depth-Anything-3 点云（任意相机系、有飞点、单位常为米）做了以下加强：

1. **Yaw 估计更稳（密度图不再倾斜）**
   - 症状：即便已有 yaw 检测，密度图仍是斜的——根因是 `estimate_yaw` 失效。
   - `estimate_yaw` 现在取**中段高度的墙面点**（`wall_band` 百分位带）而非全体点，
     避开地/顶/家具的面填充淹没薄墙信号；点先中心化、用**固定半径**做投影直方图。
   - `_projection_sharpness` 改用**固定 range `(lo, hi)`** 分箱，消除离群点使分箱变粗、
     以及旋转时包围盒增大带来的偏差。用「归一化直方图平方和」度量投影锐度（曼哈顿世界）。

2. **离群点裁剪不切角（密度图不再缺四个角）**
   - 症状：早期用轴对齐 p2/p98 方盒裁剪，在**旋转前的倾斜系**里切掉了斜放房间的四角 → 八边形。
   - 现方案（旋转不变）：
     - 高度轴：保留 `[pct_low, pct_high]` 百分位带（1D，不影响俯视轮廓，顺带去天花/地噪飞点）。
     - 楼面 (x,y)：**到中位数中心的半径**做 Tukey 栅栏 `rad ≤ Q3 + k·IQR`；半径旋转不变，永不切角，
       且各方向飞点都能抓。飞点去掉后 `generate_density` 的 min/max 自然收紧、画幅贴合。

3. **密度图对比度可调（`--density_gain`）**
   - `generate_density` 用单个最密格子归一化，几个超密格子会把墙压到暗灰。
   - 在量化成 uint8 前 `density = clip(density * gain, 0, 1)`，把峰值 1/gain 以上的格子饱和到白、抬亮墙体。
   - 注意：gain 会**改变模型输入**，可能影响预测多边形；`gain=1.0` 与原图完全等价。

4. **密度图左右镜像修复（`hflip`，`--no_floor_hflip` 关闭）**
   - 症状：y-up 点云的密度图/户型图相对实际户型图**左右镜像**（如八边形飘窗房间落到了右下而非左下）。
   - 根因：`reorder_up_axis` 把 up 轴移到第 3 列，y-up 用的 `[0,2,1]` 是**奇置换**（行列式 −1），
     翻转了楼面手性 → 投影成"从下往上"看；z-up `[0,1,2]`、x-up `[1,2,0]` 是偶置换，方向正确。
   - 修复：`floor_hflip_needed(up_axis)` 判定（仅 y-up 需要），`density_from_xyz(hflip=True)` 对成品密度图
     `np.fliplr`。**纯图像空间翻转**，不动点云/yaw/手性；`pixel_to_world(hflip=True)` 用 `255-col` 反算
     使 `world_mm` 不变（→ `polys_to_3d` 零改动仍能叠回 `.ply`）。
   - JSON `normalization.hflip` 记录该图所在帧；旧 JSON 无此字段视为未翻，`align_floorplan.py` 会把其
     多边形列 `255-col` 转到俯视帧后再对齐（本机不能跑 infer，靠这条兼容存量 JSON）。

### 相关 CLI 参数（默认值）

- `--up_axis y`：竖直轴（x/y/z）。必须与实际点云一致。
- `--rotation_deg`：手动指定 yaw；给出则跳过自动估计。`--no_align` 关闭自动校正。
- `--align_search_deg 45` / `--align_step_deg 0.5`：yaw 搜索范围与步长。
- `--pct_low 2` / `--pct_high 98`：稳健范围百分位（高度带 + 记录 floor/ceiling）。
- `--crop_iqr_k 3.0`：楼面径向 Tukey 栅栏松紧，越大保留越多（狭长房间端角被削则调大 4~5）。
- `--density_gain 1.0`：密度图对比度增益，量化前乘倍并截断到 [0,1]，建议 2~5。
- `--no_floor_hflip`：关闭 y-up 的左右翻转（默认开）。z/x-up 本就不翻。

## 反投影：户型图角点 → 原始 3D（polys_to_3d.py）

`polys_to_3d.py`（包装 `tools/polys_to_3d.sh`）读取 `_polys.json`，把每个 room 的 2D 角点
反投影回**原始点云 3D 坐标系**：撤 yaw → 补 floor/ceiling 双层高度 → 逆列置换，
输出 `_world3d.json` + `_world3d.obj`（墙面环，可与原始 `.ply` 同视图叠加）。

- floor/ceiling 由 `coords_pct_low/high[2]` 恢复：`floor = -coords_pct_high[2]`，`ceiling = -coords_pct_low[2]`（高度轴 yaw 不变），可用 `--floor/--ceiling` 覆盖。
- `--up_axis` 必须与 `infer_pointcloud.py` 当初运行时一致，否则逆置换错误。
- `hflip` **不影响本脚本**：它只翻密度图，`world_mm` 被 `pixel_to_world` 的 `255-col` 抵消后逐位不变，故 `polys_to_3d.py` 无需感知 `hflip`。

## 后处理：房间墙线对齐 / 消隙（align_floorplan.py）

`align_floorplan.py`（包装 `tools/align_floorplan.sh`）把 `infer_pointcloud.py` 独立预测、
彼此有缝且略倾斜的各房间墙线**对齐、消除房间间空隙、近轴墙拉正**，**细分被模型
合并的房间**（split，默认开），并**识别内墙门/开口**（openings，默认开）。详见
`docs/align_floorplan.md`。

- 输入：`{name}_polys.json` + 对应 `.ply`。
- 原理：用 JSON 里存储的 `applied_yaw_deg`+`min/max_coords`（+ `hflip`）重建**与多边形逐像素对齐**的密度图
  → Otsu 阈值得**墙 mask**（亮=墙）→ 并查集把近轴墙约束为共享 x/y、跨房间聚类、整簇吸附到 mask
  最密的行/列（真实墙中线）→ 相邻房间共享墙坐标，空隙消失；斜边（如八边形房间）保留；
  聚类需**吸附轴相近 + 垂直区间重叠**双条件（避免左右半楼同 y 的墙被串成一簇），
  定标时墙像素**只在簇覆盖的垂直区间内**统计（`mask[c, x0:x1]` 而非整行，避免无关墙主导），
  且用**连续性感知的去噪计数**（只算属于 ≥ `--wall_min_run 5` 连续段的像素，拒绝断线/杂物列，
  但不最大化连续段以免跳到相邻平行墙）；
  小斜边收角在**吸附前后各跑一次**（吸附把两侧墙拉正后短斜边才暴露成明显角上斜边）；
  最后三层化简：连续重复点 → 近共线/回折点（到邻点连线垂距 ≤ `--collinear_tol 2`px，
  含浅回折缺口尖点）→ 锐角尖刺（内角 < `--spike_angle_deg 60` 且开口 ≤ `--spike_max_gap 10`px）。
  默认输出俯视帧（`hflip`），旧 JSON 自动转帧（见上第 4 点）；`--no_floor_hflip` 关闭。
- **房间细分（split）**：对齐后用**贴顶带**（高度 `[--split_band_lo 0.75, --split_band_hi 0.95]`
  分位段，实测安全区 lo∈[0.65,0.82]/hi∈[0.92,0.98]，避开天花板平面；墙到顶、家具/吧台
  不到顶；门洞过梁点稀达不到阈值、与带无关，靠门洞感知缺口判据兜底）投影出第二张**结构墙 mask**
  （`_split_mask.png`），检测房间内部隔墙并递归切开（shapely 半平面相交，两子房间共享
  切割线坐标，零缝零重叠）。判据全部满足才下刀（宁漏勿误）：内部弦（线两侧各 ≥
  `--split_min_size 8`px 深房间面积，1D 腐蚀；防房间自身台阶边界墙/飘窗前沿冒充）+
  双端接界（≤ `--split_end_gap 3`px；拒独立烟道/柜线）+ 去噪覆盖率 ≥ `--split_min_cover 0.5`
  + 门洞感知缺口（洞只能是 ≤2px 噪声×1 或 `[--split_door_min 7, --split_door_max 24]`px
  门洞×1；3~6px 洞=淋浴隔断类假线，拒）+ **全高交叉验证**（同线在主 mask 去噪覆盖 ≥
  `--split_main_cover 0.4`；窗帘盒只在贴顶带、到顶衣柜前脸在低处碎掉，均拒）。
  子房间带 `split_from`（切割前下标），切割记录在 `align_info.splits`；**编号会移位**。
  切割后**再跑一遍对齐吸附**：切割线是各房间独立取的 argmax，厚墙带上会与邻房已吸附
  墙线差几像素成台阶，第二遍聚类把它们并到同一条共享墙线。
  `--no_split` 关闭。星河湾实测 13→17 房间 4 刀全对、零误切。
- **门/开口识别（openings）**：对齐+细分后，对每条墙线逐位置统计**墙面板 ±`--wall_tol`** 内
  四个高度带（`--zone_floor/low/mid/top`，占楼面→天花跨度分位；楼面/天花用直方图峰
  `estimate_floor_ceiling` 定标）+ **墙前方**中段点数，分五类 W墙/S窗台/O遮挡/D通透门洞/U无数据。
  门/开口 = 连续非 W 段。关键判据：**run 级过梁判据**（整段 top-ratio 最小值 < `--top_open_thr 0.12`
  才算真开口——实墙中段扫描空洞上方处处有墙、min top 偏高被拒；逐位置 top 会重叠，靠 run 级最小值分开）；
  floor 用**绝对计数** `--floor_min_pts` 判「被扫到过」（门洞地板比墙暗，相对阈值会误判无数据）；
  占据用**相对阈值** `--open_rel_thr 0.35`×该线墙水平（弱墙/窗帘/门框漏点按比例区分）；
  **外墙洞口默认丢弃**（`wall_is_exterior` 两侧房间栅格测试，比数房间边可靠；`--keep_exterior_openings` 保留）；
  **连通性兜底** `ensure_connectivity`（`--no_ensure_connectivity` 关闭）：某房被严格判据判成孤岛时
  按证据在其共享墙补一扇门（标 `recovered`），保证每房 ≥1 门。窗全在外墙、本版整体不输出，后续单独做。
  `--no_openings` 关闭。星河湾实测 15 门 + 1 垭口、外墙洞口 0、卧室B 边界真门由连通性补回。
- 用法：`bash tools/align_floorplan.sh [polys.json] [ply] [out_dir]`。
- 产出：`_aligned_polys.json`（另含顶层 `openings` + `openings_undecided`）、`_aligned_floorplan.png`、`_mask.png`、`_split_mask.png`（贴顶带结构 mask）、`_density_hist.png`（排序密度曲线+阈值）、`_aligned_overlay.png`（墙线叠 mask 供核对）、`_openings.png`（门/开口 + 逐位置竖直剖面分类）。floorplan/overlay 默认标注 room 序号（= `_aligned_polys.json` `rooms` 下标，两图一致；锚点用 `cv2.distanceTransform` 取离墙最远内点，L 形房间不出框），`--no_room_labels` 关闭。
- 关键旋钮：`--snap_tol`（聚类容差，px）、`--angle_tol`（近轴判定，度）、`--mask_method otsu|knee|percentile`（otsu 双峰最干净；knee=排序曲线弦距膝点，阈值更宽松、墙更连续，弱墙断线时用）、`--collapse_diag_len`（连续斜边段总长 ≤ 此值则收成直角，消切角毛刺 **并拉直近轴短边防房间错位/overlap**；**默认 20**，0=关，吸附前后各跑一次，真斜墙段更长不受影响）、`--collinear_tol`（到邻点连线垂距 ≤ 此值删点，消近共线/浅回折缺口尖点，默认 2px）、`--wall_min_run`（吸附时只有 ≥ 此长度连续段的墙像素才算墙，拒绝断线/杂物列，默认 5）、`--crop_iqr_k/--pct_low/--pct_high`（须与生成 JSON 时一致）。
- **点云→密度图的共享核心**在 `util/pointcloud.py`（`infer_pointcloud.py` 与本脚本共用，无 torch 依赖）；门/开口识别新增两个通用原语也在此：`float_pixels`（逐点浮点像素坐标，density_fixed_norm 的逐点版）、`estimate_floor_ceiling`（直方图峰定楼面/天花高度）。

## 评估：预测 vs 真值户型（eval_floorplan.py）

`eval_floorplan.py`（包装 `tools/eval_floorplan.sh`）把 `{name}_aligned_polys.json` 与 realsee
真值户型（如 `data/custom/xinghewan_floorplan/`：`room_layout.json` 逐 pano 墙线 +
`openings_gt.json` **人工标注**连通清单 + `rooms_extra.json` SVG 恢复的缺失房间）
**自动配准**后输出指标。详见 `docs/eval_floorplan.md`。

- 配准：两边都曼哈顿对齐 → 只搜 4×90°×镜像×尺度网格，栅格 mask 互相关（5cm 粗 + 1cm 精）定平移；
  拟合 scale 即点云尺度质量（星河湾中线口径 0.990 ≈ MVS 尺度偏大 1%；inner 口径会额外压低 ~1%）。
- **GT 口径 `--gt_geometry`（默认 centerline）**：`rooms_centerline.json`（墙中线多边形，
  `floorplan.json.rooms_centerline.local_path` 指定）与预测的零厚共享墙**同口径**，无半墙厚
  系统差；`inner` 切回 `room_layout.json` 内表面（仅历史对比用，IoU 有每边差半墙厚的天花板）。
  无中线文件自动回退 inner。
- 指标：逐房 IoU + 房间 P/R/F1（未匹配房带**合并/成分诊断**）、角点 P/R@0.1/0.2/0.3m
  （@0.1m 天生低：256px 密度图一像素 ≈10cm 量化）、边界 Chamfer、openings 连通性
  **strict/lenient 双层**（lenient 用 >50% 覆盖集合解析合并房间；两 GT 房并进同一预测房的
  连通标 n/e 不可检出；无 openings_gt.json 自动跳过）。
- **合并 GT 再评一轮**：预测房把多个 GT 房各覆盖 ≥95% 时判为合并（p15←客厅+餐厅+阳台B 等），
  GT union 后重算全部房间/角点/Chamfer 指标（json 键 `merged_eval`，映射逐行打印）；
  只合并 GT 侧，分割错位（卫D/过道D）不卷入。两轮差距 = 房间合并的总分代价。
- 产出：`_eval.json`（机器可读全量）+ **`_eval.txt`（控制台摘要原文）** + `_eval_overlay.png`。
- 陷阱：`room_layout.json` 的 `state=false` **不是**门窗（含家具遮挡实墙/外墙窗；`children`
  是墙线按 state 变化点的共线细分、父 state=AND(children)），门窗 GT 只能人工标注；
  预测 JSON 的 `world_mm` 字段名勿信，评估从 `pixel`+`normalization` 还原并按
  `--pred_units` 换算。
- **缺失房间恢复**：room_layout.json 缺的房间（星河湾：卫生间A/衣帽间A/阳台C）
  从页面户型图 SVG 补回——页面 JS 渲染，用无头 Chrome+puppeteer-core 点"户型图"tab 后
  dump `<svg>`（每房一个 path，毫米、y 向下）存 `floorplan.svg`；再与已知房间拟合逐轴
  仿射 + **内缩定标**（SVG 是墙中线、比 GT 内表面大 ~18%，反解 buffer(-t) 得 t≈0.096m）
  + 按 openings_gt 邻居命名，产出 `rooms_extra.json`。**`floorplan.json` 顶层 `rooms_extra`
  字段记录该文件路径**（`local_path`），`load_gt_rooms` 优先按它解析、缺省回退同目录
  `rooms_extra.json` 自动合并。
- **批量流程**：`tools/run_pipeline.sh <ply_folder> [output_root]` 逐场景预测（每场景产物写入
  `<output_root>/<scene>/`，默认 `infer_out`），再 `tools/run_eval.sh [output_root] [gt_root]`
  逐场景评估——遍历 `output_root` 下各场景子文件夹的 `*_aligned_polys.json`，按场景名匹配
  `<gt_root>/<scene>_floorplan`（默认 `gt_root=data/floorplan`）跑 `eval_floorplan.sh`，`_eval.*`
  写回该场景子文件夹；缺预测/缺 GT 的场景跳过不中断，末尾打印 evaluated/skipped/failed 汇总，
  有场景失败则退出码非零。两脚本默认目录对齐（run_pipeline 输出 = run_eval 输入 = `infer_out`）。
- 星河湾基线（21 房完整 GT，回归对比用）：**中线口径** union IoU 0.963、mean IoU 0.849、
  房间 F1 0.737（客厅+餐厅+阳台B 合并为 p15），合并轮 F1 0.882、大房 IoU 0.976；
  inner 口径 union IoU 0.900、mean IoU 0.817；openings 此前实测 strict P 1.0/R 0.45、
  **lenient F1 1.000**（现暂不评测）。

## ⚠️ 单位注意

xinghewan 等 MVS 点云是**米**不是毫米。`polys_to_3d.py --units mm` 会差 1000×，按实际单位选 `--units`（`eval_floorplan.py` 对应 `--pred_units`）。
