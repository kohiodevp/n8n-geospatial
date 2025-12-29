// Example Flutter screen: upload (sample GeoJSON) + job status polling + display imported features on map
// Dependencies to add in your app pubspec.yaml:
//   flutter_riverpod: ^2.5.0
//   flutter_map: ^6.1.0
//   latlong2: ^0.9.0
//   http: ^1.2.0
// This example uses the simple GeospatialApi (http-based) and the providers defined in this repo.

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import '../api/geospatial_api.dart';
import '../providers/parcelles_providers.dart';

class ImportAndMapExample extends ConsumerStatefulWidget {
  const ImportAndMapExample({super.key});

  @override
  ConsumerState<ImportAndMapExample> createState() => _ImportAndMapExampleState();
}

class _ImportAndMapExampleState extends ConsumerState<ImportAndMapExample> {
  String? jobId;
  String jobStatus = 'idle';
  Timer? _timer;
  bool importOk = false;

  // BBOX par défaut (autour de l’exemple)
  double xmin = 2.339;
  double ymin = 48.854;
  double xmax = 2.342;
  double ymax = 48.857;

  // Données importées (GeoJSON FeatureCollection)
  Map<String, dynamic>? importsFc;

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _importSampleGeoJson() async {
    setState(() {
      jobId = null;
      jobStatus = 'uploading';
      importOk = false;
      importsFc = null;
    });

    final api = ref.read(geospatialApiProvider);

    // Petit GeoJSON d’exemple (Polygon), SRID 4326
    const sampleGeoJson = {
      'type': 'FeatureCollection',
      'features': [
        {
          'type': 'Feature',
          'id': 'sample-1',
          'properties': {'name': 'sample_parcelle_1'},
          'geometry': {
            'type': 'Polygon',
            'coordinates': [
              [
                [2.3400, 48.8550],
                [2.3410, 48.8550],
                [2.3410, 48.8560],
                [2.3400, 48.8560],
                [2.3400, 48.8550]
              ]
            ]
          }
        }
      ]
    };

    final bytes = Uint8List.fromList(utf8.encode(jsonEncode(sampleGeoJson)));

    try {
      final res = await api.postImport(fileBytes: bytes, filename: 'sample.geojson');
      final id = res['jobId']?.toString();
      setState(() {
        jobId = id;
        jobStatus = 'queued';
      });
      if (id != null) {
        _startPolling(id);
      }
    } catch (e) {
      setState(() {
        jobStatus = 'error: $e';
      });
    }
  }

  void _startPolling(String id) {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 2), (t) async {
      final api = ref.read(geospatialApiProvider);
      try {
        final statusJson = await api.getJobStatus(id);
        final status = (statusJson != null ? statusJson['status']?.toString() : 'unknown') ?? 'unknown';
        setState(() {
          jobStatus = status;
        });
        if (status == 'success' || status == 'failed') {
          t.cancel();
          setState(() {
            importOk = (status == 'success');
          });
        }
      } catch (e) {
        // keep polling but show error state briefly
        setState(() {
          jobStatus = 'error polling: $e';
        });
      }
    });
  }

  Future<void> _loadImportsForBbox() async {
    final api = ref.read(geospatialApiProvider);
    try {
      final uri = Uri.parse('${api.baseUrl}/webhook/api/imports').replace(queryParameters: {
        'xmin': xmin.toString(),
        'ymin': ymin.toString(),
        'xmax': xmax.toString(),
        'ymax': ymax.toString(),
        'limit': '50',
        'includeTotal': 'true',
      });
      final resp = await apiHttpGet(uri.toString());
      setState(() {
        importsFc = resp;
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Erreur chargement imports: $e')));
    }
  }

  // Minimal HTTP GET helper using package:http
  Future<Map<String, dynamic>> apiHttpGet(String url) async {
    final apiKey = const String.fromEnvironment('API_KEY', defaultValue: '');
    final headers = <String, String>{};
    if (apiKey.isNotEmpty) headers['x-api-key'] = apiKey;

    final response = await http.get(Uri.parse(url), headers: headers);
    if (response.statusCode != 200) {
      throw Exception('GET failed ${response.statusCode}: ${response.body}');
    }
    final data = jsonDecode(response.body);
    if (data is Map<String, dynamic>) return data;
    if (data is List && data.isNotEmpty) {
      // n8n may wrap json sometimes; attempt to unwrap
      final first = data.first;
      if (first is Map && first['json'] is Map<String, dynamic>) {
        return first['json'] as Map<String, dynamic>;
      }
    }
    throw Exception('Unexpected response shape');
  }

  @override
  Widget build(BuildContext context) {
    final polygons = <Polygon>[];
    LatLng center = LatLng((ymin + ymax) / 2, (xmin + xmax) / 2);
    if (importsFc != null && importsFc!['type'] == 'FeatureCollection') {
      final feats = (importsFc!['features'] as List?)?.cast<Map<String, dynamic>>() ?? [];
      for (final f in feats) {
        final g = f['geometry'] as Map<String, dynamic>?;
        if (g == null) continue;
        final type = g['type'] as String?;
        if (type == 'Polygon') {
          final ring = ((g['coordinates'] as List).first as List).cast<List>();
          final pts = ring.map((c) => LatLng((c[1] as num).toDouble(), (c[0] as num).toDouble())).toList();
          polygons.add(Polygon(points: pts, color: Colors.green.withOpacity(0.2), borderColor: Colors.green, borderStrokeWidth: 1));
        } else if (type == 'MultiPolygon') {
          final mp = (g['coordinates'] as List).cast<List>();
          for (final poly in mp) {
            final ring = (poly.first as List).cast<List>();
            final pts = ring.map((c) => LatLng((c[1] as num).toDouble(), (c[0] as num).toDouble())).toList();
            polygons.add(Polygon(points: pts, color: Colors.green.withOpacity(0.2), borderColor: Colors.green, borderStrokeWidth: 1));
          }
        }
      }
      if (polygons.isNotEmpty) {
        center = polygons.first.points.first;
      }
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Import + Suivi + Carte (exemple)')),
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              ElevatedButton.icon(
                onPressed: _importSampleGeoJson,
                icon: const Icon(Icons.file_upload),
                label: const Text('Importer GeoJSON d\'exemple'),
              ),
              const SizedBox(width: 12),
              Text('Job: ${jobId ?? '-'} | Status: $jobStatus'),
            ]),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(child: _numField('xmin', xmin, (v) => xmin = v)),
              const SizedBox(width: 8),
              Expanded(child: _numField('ymin', ymin, (v) => ymin = v)),
              const SizedBox(width: 8),
              Expanded(child: _numField('xmax', xmax, (v) => xmax = v)),
              const SizedBox(width: 8),
              Expanded(child: _numField('ymax', ymax, (v) => ymax = v)),
              const SizedBox(width: 8),
              ElevatedButton(
                onPressed: importOk ? _loadImportsForBbox : null,
                child: const Text('Afficher imports'),
              ),
            ]),
            const SizedBox(height: 12),
            Expanded(
              child: FlutterMap(
                options: MapOptions(initialCenter: center, initialZoom: 14),
                children: [
                  TileLayer(urlTemplate: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', subdomains: const ['a', 'b', 'c']),
                  PolygonLayer(polygons: polygons),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _numField(String label, double value, void Function(double) onChanged) {
    return TextFormField(
      initialValue: value.toStringAsFixed(6),
      decoration: InputDecoration(labelText: label, border: const OutlineInputBorder()),
      keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: true),
      onChanged: (s) {
        final v = double.tryParse(s);
        if (v != null) onChanged(v);
      },
    );
  }
}
