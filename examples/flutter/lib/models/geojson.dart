// Minimal GeoJSON model helpers

class GeoJsonGeometry {
  GeoJsonGeometry({required this.type, required this.coordinates});
  final String type;
  final dynamic coordinates;

  factory GeoJsonGeometry.fromJson(Map<String, dynamic> json) => GeoJsonGeometry(
        type: json['type'] as String,
        coordinates: json['coordinates'],
      );
}

class GeoJsonFeature {
  GeoJsonFeature({required this.geometry, required this.properties, this.id});
  final String? id;
  final GeoJsonGeometry geometry;
  final Map<String, dynamic> properties;

  factory GeoJsonFeature.fromJson(Map<String, dynamic> json) => GeoJsonFeature(
        id: json['id']?.toString(),
        geometry: GeoJsonGeometry.fromJson(json['geometry'] as Map<String, dynamic>),
        properties: (json['properties'] as Map).cast<String, dynamic>(),
      );
}

class GeoJsonFeatureCollection {
  GeoJsonFeatureCollection({required this.features});
  final List<GeoJsonFeature> features;

  factory GeoJsonFeatureCollection.fromJson(Map<String, dynamic> json) => GeoJsonFeatureCollection(
        features: (json['features'] as List)
            .cast<Map<String, dynamic>>()
            .map(GeoJsonFeature.fromJson)
            .toList(),
      );
}
