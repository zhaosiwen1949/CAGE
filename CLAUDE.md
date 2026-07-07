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
彼此有缝且略倾斜的各房间墙线**对齐、消除房间间空隙、近轴墙拉正**。详见
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
- 用法：`bash tools/align_floorplan.sh [polys.json] [ply] [out_dir]`。
- 产出：`_aligned_polys.json`、`_aligned_floorplan.png`、`_mask.png`、`_density_hist.png`（排序密度曲线+阈值）、`_aligned_overlay.png`（墙线叠 mask 供核对）。floorplan/overlay 默认标注 room 序号（= `_aligned_polys.json` `rooms` 下标，两图一致；锚点用 `cv2.distanceTransform` 取离墙最远内点，L 形房间不出框），`--no_room_labels` 关闭。
- 关键旋钮：`--snap_tol`（聚类容差，px）、`--angle_tol`（近轴判定，度）、`--mask_method otsu|knee|percentile`（otsu 双峰最干净；knee=排序曲线弦距膝点，阈值更宽松、墙更连续，弱墙断线时用）、`--collapse_diag_len`（连续斜边段总长 ≤ 此值则收成直角，消切角毛刺 **并拉直近轴短边防房间错位/overlap**；**默认 20**，0=关，吸附前后各跑一次，真斜墙段更长不受影响）、`--collinear_tol`（到邻点连线垂距 ≤ 此值删点，消近共线/浅回折缺口尖点，默认 2px）、`--wall_min_run`（吸附时只有 ≥ 此长度连续段的墙像素才算墙，拒绝断线/杂物列，默认 5）、`--crop_iqr_k/--pct_low/--pct_high`（须与生成 JSON 时一致）。
- **点云→密度图的共享核心**在 `util/pointcloud.py`（`infer_pointcloud.py` 与本脚本共用，无 torch 依赖）。

## ⚠️ 单位注意

xinghewan 等 MVS 点云是**米**不是毫米。`polys_to_3d.py --units mm` 会差 1000×，按实际单位选 `--units`。
