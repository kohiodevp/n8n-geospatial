// Minimal Dart API client for n8n Geospatial endpoints
// Dependencies expected:
//   http: ^1.2.0
// Usage:
//   final api = GeospatialApi(baseUrl: const String.fromEnvironment('API_BASE_URL', defaultValue: 'http://localhost:5678'));
//   final feature = await api.getParcelleById('123');

import 'dart:convert';
import 'package:http/http.dart' as http;

class GeospatialApi {
  GeospatialApi({required this.baseUrl});
  final String baseUrl; // e.g. http://localhost:5678

  Uri _url(String path, [Map<String, String>? query]) =>
      Uri.parse('$baseUrl$path').replace(queryParameters: query);

  Future<Map<String, dynamic>?> getParcelleById(String id) async {
    final resp = await http.get(_url('/webhook/api/parcelles/$id'));
    if (resp.statusCode == 404) return null;
    if (resp.statusCode != 200) {
      throw HttpException('GET /parcelles/$id failed: ${resp.statusCode} ${resp.body}');
    }
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> getParcellesBbox({
    required double xmin,
    required double ymin,
    required double xmax,
    required double ymax,
    int srid = 4326,
    int limit = 100,
    int offset = 0,
  }) async {
    final resp = await http.get(_url('/webhook/api/parcelles', {
      'xmin': xmin.toString(),
      'ymin': ymin.toString(),
      'xmax': xmax.toString(),
      'ymax': ymax.toString(),
      'srid': srid.toString(),
      'limit': limit.toString(),
      'offset': offset.toString(),
    }));
    if (resp.statusCode != 200) {
      throw HttpException('GET /parcelles bbox failed: ${resp.statusCode} ${resp.body}');
    }
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> postImport({
    required List<int> fileBytes,
    required String filename,
  }) async {
    final uri = _url('/webhook/api/import');
    final req = http.MultipartRequest('POST', uri);
    req.files.add(http.MultipartFile.fromBytes('data', fileBytes, filename: filename));
    final streamed = await req.send();
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw HttpException('POST /import failed: ${resp.statusCode} ${resp.body}');
    }
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>?> getJobStatus(String jobId) async {
    final resp = await http.get(_url('/webhook/api/jobs/$jobId/status'));
    if (resp.statusCode == 404) return null;
    if (resp.statusCode != 200) {
      throw HttpException('GET /jobs/$jobId/status failed: ${resp.statusCode} ${resp.body}');
    }
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }
}

class HttpException implements Exception {
  HttpException(this.message);
  final String message;
  @override
  String toString() => message;
}
