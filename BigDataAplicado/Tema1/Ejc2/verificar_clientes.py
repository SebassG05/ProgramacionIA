"""
Verificación de restricciones para los documentos JSON de clientes
"""

import json
import os
from pathlib import Path


def load_json_files():
    """Carga todos los archivos JSON de clientes"""
    clientes = []
    script_dir = Path(__file__).parent
    
    for i in range(1, 6):
        filename = f"cliente{i}.json"
        filepath = script_dir / filename
        
        with open(filepath, 'r', encoding='utf-8') as f:
            cliente = json.load(f)
            clientes.append(cliente)
            print(f"✓ Cargado: {filename}")
    
    return clientes


def verificar_restricciones(clientes):
    """Verifica que se cumplen todas las restricciones"""
    
    print("\n" + "="*70)
    print("VERIFICACIÓN DE RESTRICCIONES")
    print("="*70)
    
    # Restricción 1: Al menos un cliente menor de 30 años
    menores_30 = [c for c in clientes if c['edad'] < 30]
    print(f"\n1. Clientes menores de 30 años: {len(menores_30)}")
    for c in menores_30:
        print(f"   - {c['nombre']}: {c['edad']} años")
    print(f"   ✓ Restricción cumplida (mínimo 1): {len(menores_30) >= 1}")
    
    # Restricción 2: Al menos dos clientes menores de 30 o de Madrid
    menores_30_o_madrid = [c for c in clientes if c['edad'] < 30 or c['ciudad'] == 'Madrid']
    print(f"\n2. Clientes menores de 30 O de Madrid: {len(menores_30_o_madrid)}")
    for c in menores_30_o_madrid:
        condicion = "< 30 años" if c['edad'] < 30 else ""
        condicion += " y " if c['edad'] < 30 and c['ciudad'] == 'Madrid' else ""
        condicion += "Madrid" if c['ciudad'] == 'Madrid' else ""
        print(f"   - {c['nombre']}: {c['edad']} años, {c['ciudad']} ({condicion})")
    print(f"   ✓ Restricción cumplida (mínimo 2): {len(menores_30_o_madrid) >= 2}")
    
    # Restricción 3: Solo tres clientes con compras > 100€
    clientes_mas_100 = []
    for c in clientes:
        total_compras = sum(compra['precio'] for compra in c['compras'])
        if total_compras > 100:
            clientes_mas_100.append((c, total_compras))
    
    print(f"\n3. Clientes con compras totales > 100€: {len(clientes_mas_100)}")
    for c, total in clientes_mas_100:
        print(f"   - {c['nombre']}: {total:.2f}€")
    print(f"   ✓ Restricción cumplida (exactamente 3): {len(clientes_mas_100) == 3}")
    
    # Restricción 4: Al menos un cliente ha comprado un "Portátil"
    clientes_portatil = []
    for c in clientes:
        for compra in c['compras']:
            if 'portátil' in compra['producto'].lower():
                clientes_portatil.append(c)
                break
    
    print(f"\n4. Clientes que compraron Portátil: {len(clientes_portatil)}")
    for c in clientes_portatil:
        portatil = [comp for comp in c['compras'] if 'portátil' in comp['producto'].lower()][0]
        print(f"   - {c['nombre']}: {portatil['producto']} ({portatil['precio']}€)")
    print(f"   ✓ Restricción cumplida (mínimo 1): {len(clientes_portatil) >= 1}")
    
    # Restricción 5: Un cliente de ciudad distinta de Madrid, Barcelona o Sevilla
    ciudades_principales = {'Madrid', 'Barcelona', 'Sevilla'}
    clientes_otras_ciudades = [c for c in clientes if c['ciudad'] not in ciudades_principales]
    
    print(f"\n5. Clientes de ciudades diferentes a Madrid/Barcelona/Sevilla: {len(clientes_otras_ciudades)}")
    for c in clientes_otras_ciudades:
        print(f"   - {c['nombre']}: {c['ciudad']}")
    print(f"   ✓ Restricción cumplida (mínimo 1): {len(clientes_otras_ciudades) >= 1}")
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    
    todas_cumplidas = (
        len(menores_30) >= 1 and
        len(menores_30_o_madrid) >= 2 and
        len(clientes_mas_100) == 3 and
        len(clientes_portatil) >= 1 and
        len(clientes_otras_ciudades) >= 1
    )
    
    if todas_cumplidas:
        print("✓ TODAS LAS RESTRICCIONES SE CUMPLEN")
    else:
        print("✗ ALGUNAS RESTRICCIONES NO SE CUMPLEN")
    
    return todas_cumplidas


def mostrar_estadisticas(clientes):
    """Muestra estadísticas generales de los clientes"""
    print("\n" + "="*70)
    print("ESTADÍSTICAS GENERALES")
    print("="*70)
    
    print(f"\nTotal de clientes: {len(clientes)}")
    
    # Edad promedio
    edad_promedio = sum(c['edad'] for c in clientes) / len(clientes)
    print(f"Edad promedio: {edad_promedio:.1f} años")
    
    # Ciudades únicas
    ciudades = set(c['ciudad'] for c in clientes)
    print(f"Ciudades representadas: {', '.join(sorted(ciudades))}")
    
    # Total de compras
    total_compras = sum(len(c['compras']) for c in clientes)
    print(f"Total de compras: {total_compras}")
    
    # Gasto total
    gasto_total = sum(
        sum(compra['precio'] for compra in c['compras'])
        for c in clientes
    )
    print(f"Gasto total: {gasto_total:.2f}€")
    print(f"Gasto promedio por cliente: {gasto_total/len(clientes):.2f}€")
    
    # Detalles por cliente
    print("\n" + "-"*70)
    print("DETALLE POR CLIENTE")
    print("-"*70)
    
    for i, c in enumerate(clientes, 1):
        total_cliente = sum(compra['precio'] for compra in c['compras'])
        print(f"\nCliente {i}: {c['cliente']}")
        print(f"  Edad: {c['edad']} años")
        print(f"  Ciudad: {c['ciudad']}")
        print(f"  Número de compras: {len(c['compras'])}")
        print(f"  Gasto total: {total_cliente:.2f}€")
        print(f"  Productos:")
        for compra in c['compras']:
            print(f"    - {compra['producto']}: {compra['precio']}€ ({compra['fecha']})")


def main():
    print("="*70)
    print("ANÁLISIS DE DOCUMENTOS JSON DE CLIENTES")
    print("="*70)
    print()
    
    # Cargar archivos JSON
    clientes = load_json_files()
    
    # Verificar restricciones
    verificar_restricciones(clientes)
    
    # Mostrar estadísticas
    mostrar_estadisticas(clientes)
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()
