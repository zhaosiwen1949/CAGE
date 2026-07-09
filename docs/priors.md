# 点云 → 户型图 重建管线：人工先验规则清单

本文件汇总从点云 `data/custom/xinghewan_da3_mvs.ply` + 初步户型图
`infer_out/xinghewan_da3_mvs_polys.json` 出发，整条重建管线里**人工设定的规则判断
（先验）**：每条规则的**原理**、对应的**代码步骤/函数**、以及控制它的**脚本参数**。

这些先验不是模型学出来的，而是基于「点云是室内场景、重力向下、墙体竖直到顶、门窗有
固定尺寸、户型是曼哈顿世界」等物理常识手工编码进代码的。调参或迁移到新场景时，先读本文。

## 管线阶段与脚本

| 阶段 | 脚本 / 函数 | 输入 → 输出 |
| --- | --- | --- |
| A. 点云预处理 + 投影密度图 | `infer_pointcloud.py` → `util/pointcloud.py` | `.ply` → 256×256 密度图 → 模型 → `_polys.json` |
| B. 密度图 → 墙 mask | `align_floorplan.py:density_to_mask` | 密度图 → 二值墙 mask |
| C. 墙线对齐 / 消隙 | `align_floorplan.py:align_rooms` | 独立房间多边形 → 共享墙、拉正 |
| D. 房间细分 | `align_floorplan.py:split_rooms` | 合并房间 → 切成真实小房间 |
| E. 门 / 开口识别 | `align_floorplan.py:detect_openings` | 墙线 → 门 / 垭口 |
| F. 反投影回 3D | `polys_to_3d.py` | 2D 角点 → 原始点云 3D 坐标 |

> 说明文档：A/F 见本仓库 `CLAUDE.md`；B–E 的算法细节见 `docs/align_floorplan.md`；
> 本文只列**先验规则**本身。

---

## A. 点云预处理与密度图投影（`infer_pointcloud.py` / `util/pointcloud.py`）

| # | 先验规则 | 原理 | 代码步骤 | 参数（默认） |
| --- | --- | --- | --- | --- |
| A1 | **竖直轴须显式指定并移到第 3 列** | 点云的"上"方向依相机系而定，投成俯视图前必须把竖直轴分离出来。y-up 用列置换 `[0,2,1]` | `reorder_up_axis` | `--up_axis y` |
| A2 | **高度按 [2,98] 百分位带裁剪** | 天花/地面外的飞点（反光、窗外）多在高度极端；1D 百分位裁剪不影响俯视轮廓，只去飞点 | `preprocess_xyz` 高度轴 | `--pct_low 2` / `--pct_high 98` |
| A3 | **楼面用「到中位中心的半径」做 Tukey 栅栏裁剪** | 轴对齐方盒会在斜放房间处切掉四角 → 八边形；**半径是旋转不变量**，永不切角，且各方向飞点都能抓 | `preprocess_xyz` 楼面 `rad ≤ Q3 + k·IQR` | `--crop_iqr_k 3.0`（狭长房间端角被削则调大 4~5） |
| A4 | **yaw 只取「中段高度的墙面点」估计** | 地/顶/家具的面填充会淹没薄墙信号；取中段高度带（`wall_band` 20~80 百分位）只留墙面点，投影直方图最锐时即楼面转正角 | `estimate_yaw` | `wall_band=(20,80)` 内部常量；`--align_search_deg 45` / `--align_step_deg 0.5`；`--rotation_deg` 手动；`--no_align` 关闭 |
| A5 | **投影锐度用「固定 range 分箱 + 归一化直方图平方和」** | 曼哈顿世界里墙平行于轴时投影最集中；固定 `(lo,hi)` 分箱消除离群点/包围盒随旋转变化带来的偏差 | `_projection_sharpness` | 同 A4 |
| A6 | **密度图按单个最密格子归一化，可选 gain 抬亮** | 少数超密格子会把墙压成暗灰；gain 把峰值 1/gain 以上饱和到白、抬亮墙体 | `density_from_xyz` 末尾 `clip(density*gain,0,1)` | `--density_gain 1.0`（建议 2~5；=1 与原图等价） |
| A7 | **y-up 密度图须左右镜像（hflip）** | `reorder_up_axis` 的 `[0,2,1]` 是**奇置换**（行列式 −1），翻了手性 → 投成"从下往上"看的镜像；`np.fliplr` 转成常规俯视 | `density_from_xyz(hflip)`；`pixel_to_world` 用 `255-col` 反算使 world 不变 | `--no_floor_hflip` 关闭（仅 y-up 需要；z/x-up 本就不翻） |

> A2/A3 的 `--pct_*/--crop_iqr_k` 决定「哪些点入图」，**下游 `align_floorplan.py` 必须用相同值**，
> 否则重建的 mask 与 JSON 多边形对不上。

---

## B. 密度图 → 墙 mask（`align_floorplan.py`）

| # | 先验规则 | 原理 | 代码步骤 | 参数（默认） |
| --- | --- | --- | --- | --- |
| B1 | **阈值只在非零密度像素上算** | 256×256 里绝大多数格子是 0（空）；含进去会退化成"0 vs 非0"而非"墙 vs 地/家具" | `density_to_mask` 取 `dens_u8>0` | 全模式共用 |
| B2 | **亮 = 墙** | 竖直墙的点在俯视投影里沿墙线密集堆叠，密度远高于地/家具 | `mask = dens ≥ thr` | — |
| B3 | **默认 Otsu 自动阈值** | 墙 vs 非墙在非零直方图上常近似双峰，Otsu 最大化类间方差最干净 | `cv2.threshold(...OTSU)` | `--mask_method otsu` |
| B4 | **弱墙断线时可换 knee（更宽松）** | 直方图偏态/单峰时 Otsu 会把弱墙切没；knee 取排序曲线弦距膝点，阈值更低、墙更连续 | `density_to_mask` knee 分支 | `--mask_method knee`；`percentile` + `--mask_percentile 80` 手动 |

---

## C. 墙线对齐 / 消除房间间空隙（`align_floorplan.py:align_rooms`）

| # | 先验规则 | 原理 | 代码步骤 | 参数（默认） |
| --- | --- | --- | --- | --- |
| C1 | **与轴夹角 ≤ 阈值的墙拉成严格水平/竖直，其余保留为斜边** | 户型墙绝大多数是轴对齐的，几度偏差是模型噪声；但八边形飘窗等真斜墙不能拉直 | `_classify_edge` 返回 H/V/D | `--angle_tol 8.0`（真斜墙被拉直→调小 5） |
| C2 | **同轴墙坐标做并查集聚类 → 相邻房间共享一条墙线** | 相邻房间各自预测的墙相差几像素、留缝；把近邻坐标归一即消隙 | `align_rooms` 并查集 + `_cluster_and_snap` | `--snap_tol 5.0`（缝没消干净→调大；误并→调小） |
| C3 | **聚类需「吸附轴相近 **且** 垂直区间重叠」双条件** | 只按坐标聚会把左右半楼同 y 的墙串成一簇，把吸附点拖到墙多的一侧 | `_cluster_and_snap` 的 `gap` 判据 | `--snap_tol`（同时用于两条件） |
| C4 | **定标只在簇覆盖的垂直区间内统计墙像素** | 一段共享墙只占有限 x 范围；统计整行会让别处无关墙主导 argmax | `_cluster_and_snap` 用 `mask[c, p0:p1]` 而非整行 | 由簇 span 自动决定 |
| C5 | **墙像素用「连续性感知去噪计数」** | 断断续续的杂物列（家具/管井）能凑出与真墙相同的原始计数；只算 ≥ min_run 连续段的像素来拒绝它，但不最大化连续段（否则会跳到相邻平行墙） | `_wall_score` | `--wall_min_run 5`（吸到断线列→调大 6~8） |
| C6 | **短连续斜边段收成直角** | 模型在墙角会幻觉出小切角（八边形毛刺）；近轴短边略超 angle_tol 会被判斜边、不吸附而错位/overlap。总长 ≤ 阈值的斜边段收成直角，真斜墙段更长不受影响 | `_collapse_short_diagonals`（吸附前后各跑一次） | `--collapse_diag_len 20`（0=关；真斜墙段 ≥ 32px） |
| C7 | **化简层1：近共线/浅回折点删除** | 到邻点连线垂距 ≤ 阈值的点是共线噪声或浅回折缺口尖点，删掉outline只动 ≤ 该值 | `_simplify_polygon` 垂距判据 | `--collinear_tol 2`（浅折角被拉平→调小/设0） |
| C8 | **化简层2：锐角窄口尖刺删除** | **户型图不出现锐角**；内角 < 阈值 **且** 开口窄（守卫真实宽楔形）的顶点是针状尖刺 | `_simplify_polygon` 角度+开口判据 | `--spike_angle_deg 60` / `--spike_max_gap 10`（0=关） |

---

## D. 房间细分（`align_floorplan.py:split_rooms`）

核心先验：**墙体到顶、家具/门扇不到顶**，所以另投一张「贴顶带」结构墙 mask 找房间内部隔墙。

| # | 先验规则 | 原理 | 代码步骤 | 参数（默认） |
| --- | --- | --- | --- | --- |
| D1 | **隔墙证据取「贴顶带」（高度 [0.75,0.95] 分位段）** | 墙到顶=贴顶带里连续亮线；家具/吧台/门扇不到顶→消失；门洞被过梁封（若重建出） | `ceiling_band_mask` | `--split_band_lo 0.75` / `--split_band_hi 0.95`（安全区 lo∈[0.65,0.82]、hi∈[0.92,0.98]） |
| D2 | **贴顶带上界须 < 1.0（避开天花板平面本体）** | 天花板是水平面，含进去会整片泛白淹没内墙 | `ceiling_band_mask` 用高度 min/max 的分位 | `--split_band_hi ≤ 0.98` |
| D3 | **贴顶带阈值用百分位而非 Otsu** | 贴顶带直方图不双峰，Otsu 不可靠 | `density_to_mask(method='percentile')` | `--split_mask_percentile 80` |
| D4 | **下刀判据一：内部弦（线两侧各 ≥ min_size 深房间面积）** | 防房间自身台阶边界墙 / 飘窗前沿冒充隔墙（它们骑在边界墙厚度里） | `_best_split_line_axis` 1D 腐蚀 `region_int` | `--split_min_size 8`（同时是子房间最小边长） |
| D5 | **下刀判据二：双端接界（首/末墙像素距弦端 ≤ 阈值）** | 真隔墙两端都锚在房间边界；独立烟道/高柜/吧台线最多一端搭墙 | `_best_split_line_axis` 端距判据 | `--split_end_gap 3` |
| D6 | **下刀判据三：去噪覆盖率 ≥ 阈值** | 隔墙须近满跨房间；碎线不算 | `_denoised_cover`（复用 `--wall_min_run`） | `--split_min_cover 0.5` |
| D7 | **下刀判据四：门洞感知缺口** | 线上空洞只能是噪声（≤2px×1）或门洞尺寸（[7,24]px×1）；3~6px 洞=淋浴隔断类假线（现实墙上不存在这么窄的洞） | `_best_split_line_axis` gap 分类 | `--split_door_min 7` / `--split_door_max 24`；内部常量 `_SPLIT_NOISE_GAP=2` / `_SPLIT_MAX_NOISE_GAPS=1` |
| D8 | **下刀判据五：全高交叉验证（主 mask 也够亮）** | 真隔墙从地到顶、两张 mask 都亮；窗帘盒只在贴顶带、到顶衣柜前脸在低处被杂物打碎 | `_best_split_line_axis` 查 `main_mask` | `--split_main_cover 0.4`（0=关） |
| D9 | **切割用 shapely 半平面相交，两子房间共享切割线坐标** | 与共享墙约定一致，零缝零重叠 | `_cut_polygon` | — |
| D10 | **切割后再跑一遍对齐吸附** | 切割线是各房间独立取的 argmax，厚墙带上会与邻房已吸附墙线差几像素成台阶；第二遍聚类并到同一条 | `main` 中 split 后再调 `align_rooms` | 复用 C 的参数 |

> 判据全部满足才下刀（宁漏勿误）。星河湾实测 13→17 房间、4 刀全对、零误切。
> `--no_split` 关闭。子房间带 `split_from`（切割前下标），**编号会移位**。

---

## E. 门 / 开口识别（`align_floorplan.py:detect_openings`）

核心先验：2D mask 分不清「门洞」和「被遮挡/没扫到的实墙」（都在中段空）；改用**每个墙位置
的竖直点分布**判定。楼面/天花高度用直方图峰定标（`estimate_floor_ceiling`），高度带按占
楼面→天花跨度的分位表达。

| # | 先验规则 | 原理 | 代码步骤 | 参数（默认） |
| --- | --- | --- | --- | --- |
| E1 | **逐位置五分类 W墙/S窗台/O遮挡/D门洞/U无数据** | 门=墙面板中低带空且**地板可见**；窗台=低带有中带空；遮挡=墙面板空但前方有家具；无数据=处处无点 | `classify_positions` | 高度带 `--zone_floor/low/mid/top`（[-.08,.18]/[.20,.33]/[.40,.72]/[.78,.96]）；墙面板半宽 `--wall_tol 1.5`；前方 `--front_tol 5.0` |
| E2 | **占据判定用「相对该线自身墙水平」的阈值** | 弱墙/窗帘/门框漏点的绝对计数各不相同；按该线中带 75 分位的比例判据才稳 | `classify_positions` `rel_thr·wall_level` | `--open_rel_thr 0.35`；`--open_min_pts 5`；`--open_min_wall_dens 60` |
| E3 | **地板「是否被扫到」用绝对计数** | 门洞地板比墙暗，用相对阈值会把真门误判成无数据 U | `classify_positions` `floor_scanned` | `--floor_min_pts 5` |
| E4 | **run 级过梁判据：整段 top-ratio 最小值 < 阈值才算真开口** | 真门**某处开到天花**（MVS 不重建门楣）；实墙的中段扫描空洞**上方处处有墙**（min top-ratio 偏高）。逐位置 top 会重叠，靠整段最小值分开（实测真门触底 0.08、实墙空洞 ≥0.16） | `detect_openings` 洞循环 `top_ratio[s:e].min()` | `--top_open_thr 0.12`（实墙冒假门→调小；真门漏→调大） |
| E5 | **遮挡段（O 占多数）不算开口** | 家具挡在墙前造成中段空，但那是墙不是门 | `detect_openings` `frac['O']≥0.5` 跳过 | 由 E1 的 `front` 计数决定 |
| E6 | **门/开口按宽度分：[7,24]px 为门，更宽为垭口** | 门有固定尺寸（~0.7~2.2m）；更宽的无墙段是开放垭口 | `detect_openings` kind 判定 | `--door_min 7` / `--door_max 24`；`--open_hole_min 6` 最小上报宽 |
| E7 | **外墙洞口默认全部丢弃（本版只留内墙门/开口）** | 户型内墙无窗；外墙的门/窗/开口本版不做（窗全在外墙，后续单独做）。外墙判定用**两侧房间栅格**测试（比数房间边可靠：两条房间边并到一线也能判对） | `wall_is_exterior` + `detect_openings` 过滤 | `--keep_exterior_openings` 保留 |
| E8 | **户型内墙无窗：内墙上「窗台型」洞降级为门** | 内墙不会有窗；低带那点是门后紧贴的矮柜/洗手台 | `detect_openings` 内墙 S→door | — |
| E9 | **纯无数据（U）内墙段不算门，只在调试图标灰虚线** | 没扫到 ≠ 有洞；不假装是门也不假装是墙 | `detect_openings` `frac['U']≥0.6`→undecided | — |
| E10 | **连通性兜底：每个 room 至少一扇门连到别的 room** | 严格判据偶尔漏掉边界真门致某房成孤岛（物理进不去）；在其共享墙上按证据强度补一扇门（标 `recovered`） | `ensure_connectivity` | `--no_ensure_connectivity` 关闭 |

> `--no_openings` 关闭。星河湾实测 15 门 + 1 垭口（餐厅→厨房）、外墙洞口 0、卧室B 边界真门
> 由连通性补回、位置正确。

---

## F. 反投影回 3D（`polys_to_3d.py`）

| # | 先验规则 | 原理 | 代码步骤 | 参数（默认） |
| --- | --- | --- | --- | --- |
| F1 | **floor/ceiling 双层高度由 `coords_pct_low/high[2]` 恢复** | `ps` 帧里 `ps_z = -height`，故 `floor = -高分位 z`、`ceiling = -低分位 z`；百分位版排除了地噪/家具顶 | `recover_height_range` | `--floor` / `--ceiling` 覆盖 |
| F2 | **`--up_axis` 必须与 infer 当初一致** | 逆列置换要与正向置换互逆，否则 x/y/z 错位 | `inverse_perm` | `--up_axis`（默认读 JSON 记录） |
| F3 | **hflip 不影响本脚本** | hflip 只翻密度图，`world_mm` 被 `pixel_to_world` 的 `255-col` 抵消后逐位不变 | — | — |
| F4 | **⚠ 单位：MVS 点云是米不是毫米** | xinghewan 等 MVS 点云单位为米；`--units mm` 会差 1000× | `process_file` 读 `world_mm`/`world_m` | `--units mm\|m`（按实际单位选） |

---

## 跨阶段一致性红线

- **A2/A3 的 `--pct_low/--pct_high/--crop_iqr_k` 必须在 `infer_pointcloud.py` 与
  `align_floorplan.py` 之间保持一致**，否则重建密度图与 JSON 多边形错位（B–E 全部失真）。
- **点云→密度图的共享核心**在 `util/pointcloud.py`（`reorder_up_axis / preprocess_xyz /
  estimate_yaw / rotate_floor_plane / density_fixed_norm / float_pixels /
  estimate_floor_ceiling / pixel_to_world`），`infer_pointcloud.py` 与 `align_floorplan.py`
  共用，无 torch 依赖 —— 保证正向投影与后处理重建逐像素一致。
- 所有"宁严"取向（C5/C6、D4–D8、E4/E7）都遵循同一原则：**宁可漏检也不误检**，
  漏的由更宽松参数或连通性兜底（E10）补回。
