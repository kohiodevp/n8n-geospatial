// Example Widget using flutter_map to display GeoJSON FeatureCollection
// Add to pubspec.yaml:
// dependencies:
//   flutter_map: ^6.1.0
//   latlong2: ^0.9.0
//   flutter_riverpod: ^2.5.0
//   http: ^1.2.0
//
// This is a minimal example; in a real app, handle errors/loading states gracefully.

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/parcelles_providers.dart';

class GeoJsonMapExample extends ConsumerWidget {
  const GeoJsonMapExample({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final query = BboxQuery(
      xmin: 2.33, ymin: 48.85, xmax: 2.35, ymax: 48.86, limit: 200,
    );
    final async = ref.watch(parcellesBboxProvider(query));

    return Scaffold(
      appBar: AppBar(title: const Text('Parcelles - GeoJSON (bbox)')),
      body: async.when(
        data: (fcoll) {
          final polygons = <Polygon>[];
          for (final f in fcoll.features) {
            if (f.geometry.type == 'Polygon') {
              final coords = (f.geometry.coordinates as List)
                  .first // exterior ring
                  .cast<List>()
                  .map((c) => LatLng(c[1] as num? ?? 0.0, c[0] as num? ?? 0.0))
                  .toList();
              polygons.add(Polygon(points: coords, color: Colors.blue.withOpacity(0.2), borderColor: Colors.blue, borderStrokeWidth: 1));
            } else if (f.geometry.type == 'MultiPolygon') {
              final mp = (f.geometry.coordinates as List).cast<List>();
              for (final poly in mp) {
                final ring = poly.first.cast<List>().map((c) => LatLng(c[1] as num? ?? 0.0, c[0] as num? ?? 0.0)).toList();
                polygons.add(Polygon(points: ring, color: Colors.blue.withOpacity(0.2), borderColor: Colors.blue, borderStrokeWidth: 1));
              }
            }
          }
          final center = LatLng((query.ymin + query.ymax) / 2, (query.xmin + query.xmax) / 2);
          return FlutterMap(
            options: MapOptions(initialCenter: center, initialZoom: 14),
            children: [
              TileLayer(urlTemplate: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', subdomains: const ['a','b','c']),
              PolygonLayer(polygons: polygons),
            ],
          );
        },
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, st) => SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Text('Erreur: $e\n\n$st'),
        ),
      ),
    );
  }
}
