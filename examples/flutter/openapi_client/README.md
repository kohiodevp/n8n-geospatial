# geospatial_api_client (OpenAPI)

Client Dart généré via OpenAPI Generator (dart-dio-next).

## Génération

- Prérequis: Docker
- Commande:
  - `bash scripts/generate_dart_client.sh`
- Le client est généré dans `examples/flutter/openapi_client`

## Utilisation (Dio + x-api-key)

```dart
import 'package:dio/dio.dart';
import 'package:geospatial_api_client/api.dart';
import 'package:geospatial_api_client/auth_api_key_interceptor.dart';

void main() async {
  final dio = Dio()
    ..options.baseUrl = const String.fromEnvironment('API_BASE_URL', defaultValue: 'http://localhost:5678')
    ..interceptors.add(ApiKeyInterceptor(const String.fromEnvironment('API_KEY', defaultValue: '')));

  final api = GeospatialApi(dio: dio);

  // Exemple: GET parcelles par bbox
  final resp = await dio.get('/webhook/api/parcelles', queryParameters: {
    'xmin': 2.33,
    'ymin': 48.85,
    'xmax': 2.35,
    'ymax': 48.86,
    'limit': 50,
    'includeTotal': true,
  });
  print(resp.data);
}
```

## Avec Flutter + Riverpod

- Vous pouvez remplacer les appels HTTP manuels par les méthodes générées si vous mappez les endpoints.
- Ajoutez l'interceptor `ApiKeyInterceptor` pour injecter le header `x-api-key`.

## Régénération / MAJ de l'API

- Après modification de `api/openapi.json`, régénérez le client.
- Formatez le code: `dart format .` (facultatif)
