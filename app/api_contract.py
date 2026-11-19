"""Expose middleware authorization requirements in generated OpenAPI."""
from fastapi.openapi.utils import get_openapi


def install(app):
    def schema():
        if app.openapi_schema:
            return app.openapi_schema
        from app.security import required_role, context_tester_route_allowed
        value = get_openapi(title=app.title, version=app.version,
                            description=app.description, routes=app.routes)
        value.setdefault('components', {}).setdefault('securitySchemes', {})['PrivateBearer'] = {
            'type': 'http', 'scheme': 'bearer',
            'description': 'Persistent credential. Stable principal IDs own private data; never place tokens in URLs.'}
        for path, methods in value['paths'].items():
            for method, operation in methods.items():
                if method not in {'get', 'post', 'put', 'patch', 'delete', 'head', 'options'}:
                    continue
                role = required_role(path, method.upper())
                if path.startswith('/api/context/') and role:
                    operation['x-context-tester-allowed'] = context_tester_route_allowed(path, method.upper())
                if role:
                    operation['security'] = [{'PrivateBearer': []}]
                    operation['x-minimum-role'] = role
                    if path.startswith('/api/adaptive'):
                        route = next((r for r in app.routes if getattr(r, 'path', None) == path and
                                      method.upper() in getattr(r, 'methods', set())), None)
                        deps = getattr(getattr(route, 'dependant', None), 'dependencies', [])
                        if any(getattr(d.call, '__name__', '') == 'reviewer' for d in deps):
                            operation['x-minimum-role'] = 'reviewer'
                    for code, description in [('401', 'Missing or invalid private credential'),
                                              ('403', 'Role does not permit this operation'),
                                              ('503', 'Access configuration or operation unavailable')]:
                        operation.setdefault('responses', {}).setdefault(code, {'description': description})
                if method in {'post', 'put', 'patch'}:
                    operation['x-upload-deadline-seconds'] = 30
                    operation['x-upload-limit-bytes'] = (90 if path.startswith('/api/adaptive/captures') else 16)*1024**2
                    operation.setdefault('responses', {}).setdefault('408', {'description': 'Upload deadline exceeded'})
                    operation['responses'].setdefault('413', {'description': 'Streaming body limit exceeded'})
        value['x-limitations'] = ['Some legacy responses are not yet fully typed.',
                                 'WebSocket transport is lifecycle-only and is not a recognition decoder.']
        app.openapi_schema = value
        return value
    app.openapi = schema
