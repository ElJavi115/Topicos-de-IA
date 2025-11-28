import 'auto_model.dart';
import 'user_model.dart';

class PlateData {
  final Auto? autoData;
  final Persona userData;

  PlateData({
    required this.autoData,
    required this.userData,
  });

  factory PlateData.fromJson(Map<String, dynamic> json) {
    final autoJson = json['auto'];

    return PlateData(
      autoData: autoJson != null
          ? Auto.fromJson(autoJson as Map<String, dynamic>)
          : null,
      userData: Persona.fromJson(json['persona'] as Map<String, dynamic>),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'auto': autoData?.toJson(),
      'persona': userData.toJson(),
    };
  }
}
