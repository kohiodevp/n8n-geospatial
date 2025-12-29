// Riverpod providers for Parcelles API usage
// Dependencies expected:
//   flutter_riverpod: ^2.5.0
//   http: ^1.2.0

import 'dart:typed_data';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../api/geospatial_api.dart';
import '../models/geojson.dart';

defaultApiBaseUrl() => const String.fromEnvironment('API_BASE_URL', defaultValue: 'http://localhost:5678');

final geospatialApiProvider = Provider<GeospatialApi>((ref) {
  return GeospatialApi(baseUrl: defaultApiBaseUrl());
});

final parcelleByIdProvider = FutureProvider.family<GeoJsonFeature?, String>((ref, id) async {
  final api = ref.watch(geospatialApiProvider);
  final json = await api.getParcelleById(id);
  if (json == null) return null;
  return GeoJsonFeature.fromJson(json);
});

class BboxQuery {
  BboxQuery({required this.xmin, required this.ymin, required this.xmax, required this.ymax, this.srid = 4326, this.limit = 100, this.offset = 0});
  final double xmin, ymin, xmax, ymax;
  final int srid, limit, offset;
}

final parcellesBboxProvider = FutureProvider.family<GeoJsonFeatureCollection, BboxQuery>((ref, q) async {
  final api = ref.watch(geospatialApiProvider);
  final json = await api.getParcellesBbox(
    xmin: q.xmin, ymin: q.ymin, xmax: q.xmax, ymax: q.ymax, srid: q.srid, limit: q.limit, offset: q.offset,
  );
  return GeoJsonFeatureCollection.fromJson(json);
});

final importJobProvider = FutureProvider.family<Map<String, dynamic>, ({Uint8List bytes, String filename})>((ref, data) async {
  final api = ref.watch(geospatialApiProvider);
  final res = await api.postImport(fileBytes: data.bytes, filename: data.filename);
  return res; // contains jobId
});

final jobStatusProvider = FutureProvider.family<Map<String, dynamic>?, String>((ref, jobId) async {
  final api = ref.watch(geospatialApiProvider);
  return api.getJobStatus(jobId);
});
