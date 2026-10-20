"""Functions working on a stored scene (detection first). The app is the hub: it keeps the scene and
hands it over. The exchange protocol with the detector is not agreed yet, so this is an explicit stub:
nothing is sent and no detections are invented. The bundle is ready at /api/scenes/<id>/bundle.
"""
import os
from . import scenes
from .providers import ProviderUnavailable

def health():
    url=os.environ.get('ROOMPLAN_DETECT_URL','').strip()
    return {'available':False,'mode':'stub','method':'scene bundle -> external detector','status':'exchange protocol not agreed yet'+('; ROOMPLAN_DETECT_URL is set but unused' if url else '')}

def detect(sid):
    if scenes.summary(sid) is None: raise ValueError('Сцена не найдена на сервере')
    # Seam for the agreed protocol: send scenes.bundle_path(sid), wait for the detector, return its boxes.
    raise ProviderUnavailable(f'Детекция по сцене пока не подключена: протокол обмена с детектором ещё согласуется. Пакет сцены готов: /api/scenes/{sid}/bundle')
