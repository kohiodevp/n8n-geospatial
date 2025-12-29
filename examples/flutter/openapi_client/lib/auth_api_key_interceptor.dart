import 'package:dio/dio.dart';

/// Simple Dio interceptor to inject x-api-key header for protected endpoints.
class ApiKeyInterceptor extends Interceptor {
  ApiKeyInterceptor(this.apiKey);
  final String apiKey;

  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) {
    if (apiKey.isNotEmpty) {
      options.headers['x-api-key'] = apiKey;
    }
    super.onRequest(options, handler);
  }
}
