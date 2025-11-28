import 'package:plate_detector/models/plate_model.dart';

class OcrResult {
  final String textoCrudo;
  final String placaNormalizada;
  final double? score;
  final PlateData? match;

  OcrResult({
    required this.textoCrudo,
    required this.placaNormalizada,
    this.score,
    this.match,
  });
}