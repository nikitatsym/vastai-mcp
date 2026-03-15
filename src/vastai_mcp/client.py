import httpx

from .config import get_settings


class APIError(Exception):
    def __init__(self, status: int, method: str, path: str, body):
        self.status = status
        self.method = method
        self.path = path
        self.body = body
        super().__init__(f"{method} {path} -> {status}: {body}")


class VastClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        s = get_settings()
        self._base = (base_url or s.vastai_url).rstrip("/")
        self._run_base = s.vastai_run_url.rstrip("/")
        self._key = api_key or s.vastai_api_key
        headers = {"Authorization": f"Bearer {self._key}"}
        self._http = httpx.Client(
            base_url=self._base,
            headers=headers,
            timeout=30.0,
        )
        self._run_http = httpx.Client(
            base_url=self._run_base,
            headers=headers,
            timeout=30.0,
        )

    def _handle(self, r: httpx.Response):
        if r.status_code >= 400:
            try:
                body = r.json()
            except Exception:
                body = r.text
            raise APIError(r.status_code, r.request.method, str(r.url), body)
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    def get(self, path: str, **kwargs):
        return self._handle(self._http.get(path, **kwargs))

    def post(self, path: str, **kwargs):
        return self._handle(self._http.post(path, **kwargs))

    def put(self, path: str, **kwargs):
        return self._handle(self._http.put(path, **kwargs))

    def delete(self, path: str, **kwargs):
        return self._handle(self._http.delete(path, **kwargs))

    def run_post(self, path: str, **kwargs):
        """POST to run.vast.ai (serverless runtime API)."""
        return self._handle(self._run_http.post(path, **kwargs))

    def run_get(self, path: str, **kwargs):
        """GET from run.vast.ai (serverless runtime API)."""
        return self._handle(self._run_http.get(path, **kwargs))

