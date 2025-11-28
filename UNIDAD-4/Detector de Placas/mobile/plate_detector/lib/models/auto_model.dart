class Auto {
  final int id;
  final String placa;
  final String marca;
  final String modelo;
  final String color;

  Auto({
    required this.id,
    required this.placa,
    required this.marca,
    required this.modelo,
    required this.color,
  });

  factory Auto.fromJson(Map<String, dynamic> json) {
    return Auto(
      id: json['id'],
      placa: json['placa'],
      marca: json['marca'],
      modelo: json['modelo'],
      color: json['color'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'placa': placa,
      'marca': marca,
      'modelo': modelo,
      'color': color,
    };
  }
}
