# RoomPlan — prototype acceptance contract

User goal: web application for apartment/room modelling, furniture planning and realistic visualization; extensible research platform for reconstruction and neural rendering. Repository: https://github.com/antonkonushin/roomplan. Python owns model, validation, reconstruction, persistence and exports; Three.js owns interactive visualization. Architecture otherwise discretionary.

## Required scenarios
1. Import a photographed/scanned floor plan: actual baseline recognition, user ceiling height and scale calibration. Report limits and allow correction; never substitute the demo silently. Import phone walkthrough video: real plan from the external roomplan-app reconstruction service when configured (see API below), otherwise the labelled demo stub. Import multiple apartment photos (or video without a configured service): explicitly labelled deterministic demo reconstruction stubs with furnished default plan, synthetic point cloud, documented integration seams (future height/furniture recognition).
2. Orbit/pan/zoom 3D and top-down 2D, furniture/appliances visible in both. Toggle architecture, furniture, point cloud, additional research layers and object 3D boxes independently.
3. Select walls, edit length/height, propagate to model coherently.
4. Categorized furniture gallery with two references per category; floor, supported (TV on cabinet) and wall mounting; move along support surfaces.
5. Custom furniture from uploaded photographs; connect an existing single-image reconstruction method for baseline, future multi-view seam. Clearly distinguish unavailable inference from generated geometry. Do not pretend proxy meshes are AI reconstruction.
6. Save/load a portable project on user's computer including all original images/videos and reconstructed/custom assets, preserving geometry and settings.
7. Generate downloadable apartment walkthrough video using ordinary computer graphics; configurable camera path. Future generative video seam, not required now.
8. Export an attractive furnished apartment plan suitable for property listings. Ordinary Three.js raster rendering is sufficient for stills and walkthroughs. Per user clarification on 2026-10-16, ray tracing and photorealism are outside this prototype; photorealism belongs to a future separate generative-model module.

## Quality loop
Use fresh-context builder/critic agents for independently judgeable pieces. Critics inspect real running output and exports, compare with fixed Floorplanner references; anonymous randomized A/B where practical. Record biggest gap, iterate and retest integrated product. No false wins or arbitrary round cap. Maintain live /progress page with history, evidence, outstanding gaps and next actions.

References: https://floorplanner.com/2d-floorplan-examples ; https://floorplanner.com/3d-floorplan-examples ; https://floorplanner.com/3d-render-examples . The dated user-scope amendment in quality-bar.md governs active comparisons: attractive listing plans and clear, coherent ordinary-CG views, without a photorealism gate. Functionality without a reference counterpart is checked against this contract. Completion requires all mandatory scenarios and independent checks within this scope; partial completion must remain labelled partial.

## Shared implementation interface (v1)
This interface coordinates agents, not an immutable architecture.
Python stdlib HTTP server + Pillow/numpy, browser ES modules vendored Three.js. Start `python server.py --port 8765`. No build required.
All geometry metres. Scene coordinates x,z horizontal, y up. Polygon points [x,z]. Furniture position [x,y,z] is bottom centre. Rotation radians around y.
Project JSON:
`{version:1,name,ceilingHeight,walls:[{id,a:[x,z],b:[x,z],height,thickness}],rooms:[{id,name,polygon:[[x,z]],color}],objects:[{id,catalogId,name,position:[x,y,z],rotation,size:[w,h,d],color,mount:'floor'|'wall'|'surface',assetId?}],pointCloud:[[x,y,z]],layers:{},sources:[{id,name,mime,data:base64}],assets:[{id,name,mime,data:base64}],provenance:{}}`.
Catalog entries `{id,name,category,kind,size:[w,h,d],color,mount}`. kind controls browser procedural furniture geometry. Version schema must validate finite bounded coordinates and array sizes.

API routes:
- GET /api/demo -> project (furnished ~65m² apartment)
- GET /api/catalog -> catalog array
- GET /api/health -> status/providers
- POST /api/reconstruct `{mode:'plan'|'video'|'photos',files:[{name,mime,data}],ceilingHeight,widthMeters}` -> `{project,warnings}`. data is plain base64, not URL.
- POST /api/validate `{project}` -> `{project,warnings}` or 400 `{error}`
- POST /api/export/project `{project}` -> ZIP binary; POST /api/import/project `{data:base64}` -> `{project,warnings}`
- POST /api/export/plan `{project}` -> SVG binary
- POST /api/reconstruct/video `{files:[{name,mime,data}]}` (one video ≤100 MB) -> `{job:{id,status,stage,message,error,created,claimed,finished,video:{name,size}}}`; 503 when `ROOMPLAN_VIDEO_API_URL`/`ROOMPLAN_VIDEO_API_TOKEN` are unset, the URL is plain `http` to a non-loopback host (remote service: `https` with certificate verification only), or the service is unreachable. The video is forwarded to the roomplan-app API (`POST /api/jobs`, `include_cloud=true`).
- GET /api/reconstruct/video/{jobId} -> `{job}` while `queued|running`, `{job}` with `status:'failed', error` on failure, `{job,project,warnings}` when `done`. Stateless relay: the browser keeps the uploaded video and attaches it to `project.sources`. Conversion: plan rotated to the dominant wall direction, service Z-up → project y-up (`[x,y,z]→[x,z−floor,−y]` after rotation, shifted to the origin); floor/ceiling from cloud density peaks (fallback: wall bounds); TUN3D objects → catalog kinds with detected sizes (`detection` field); doors/windows → `openings` with `wallId`; cloud ≤60k points, layer `pointCloud` off by default; raw `result.json` kept as asset; transform in `provenance.alignment`.
- Scenes (shared space for functions). When a finished video job lists `scene.tar.gz` (format `roomplan-scene/1`: `images/`, `poses.json` with `K` and per-frame `T_cw` world→camera OpenCV + `checks`, `cloud.ply`, `plan.json`; one world, metres, Z up, floor z=0), the server fetches it once, validates it (single top dir, no links, no `..`, known format) and keeps it in `ROOMPLAN_DATA_DIR/scenes/<jobId>/`. The project gets `scenes:[{id,format,frames,images,cloudPoints,hasPlan,checksOk,bytes,sha256,source}]`; the portable ZIP carries `scenes/<id>.tar.gz` and import restores it.
  - GET /api/scenes/{id} -> summary; GET /api/scenes/{id}/bundle -> tar.gz.
  - POST /api/scenes/{id}/detect -> explicit stub: 503 until the detector exchange protocol is agreed (`roomplan/functions.py`); nothing is sent, no boxes are invented.
- POST /api/reconstruct/object `{files:[...],name}` -> `{asset,object,warnings}` or explicit 503 when provider unavailable. Actual model adapter may use configurable external local service, preserving uploaded inputs. Frontend can offer an explicitly labelled manual photo-proxy separately, with user dimensions.
Static frontend under /web; root serves web/index.html; /progress serves web/progress.html; /progress.json serves progress.json. All API errors JSON.
Frontend can keep source files base64 in project for archive roundtrip. Retain arbitrary research assets. Server listens loopback by default; deployment documented separately.

### Research overlays (optional project extension, layer schema v1)

`researchLayers:[{version:1,id,name,kind:'points'|'lines',visible:boolean,units:'m',coordinateFrame:'project-x-y-z',positions:[[x,y,z]],segments?:[[index,index]],color:'#RRGGBB',sourceIds:[],assetIds:[],provenance:{}}]`.

Each layer is independently visible under the `layers.research` master switch. Lines require explicit segment indices; points omit them. The backend rejects unknown kinds/versions, invalid references, duplicate layer IDs, nonfinite/out-of-bounds coordinates, and excessive counts. Original imported research JSON is retained as an asset referenced by the parsed layers and preserved by the portable ZIP. These are real user-data overlays, distinct from CG camera path helpers. See `docs/RESEARCH_LAYERS.md` for limits, source-unit requirements, and the import protocol.
