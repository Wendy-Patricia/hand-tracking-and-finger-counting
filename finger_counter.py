
import cv2 as cv

from hand_geometry import (
    extract_hand_contour,
    analyze_hand_geometry
)


def main():
    # 1. Carregar a imagem da mao
    image = cv.imread("tests/hand_5.png")

    if image is None:
        print("Erro: imagem nao encontrada.")
        return

    # 2. Extrair o contorno da mao
    contour, mask = extract_hand_contour(image)

    if contour is None:
        print("Nenhuma mao encontrada.")
        return

    # 3. Calcular Convex Hull e Convexity Defects
    hull, defects = analyze_hand_geometry(contour)

    # Criar copia da imagem para desenhar
    result = image.copy()

    # 4. Desenhar o contorno da mao - VERDE
    cv.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    # 5. Desenhar o Convex Hull - AZUL
    if hull is not None:
        cv.drawContours(
            result,
            [hull],
            -1,
            (255, 0, 0),
            2
        )

    # 6. Desenhar os Convexity Defects - VERMELHO
    if defects is not None:

        print("Shape dos defects:", defects.shape)

        # Garantir que cada linha tem 4 valores:
        # start, end, far, depth
        defects = defects.reshape(-1, 4)

        for s, e, f, depth in defects:

            # Converter indices para inteiros
            s = int(s)
            e = int(e)
            f = int(f)

            # Coordenadas dos pontos
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Desenhar o ponto mais profundo
            cv.circle(
                result,
                far,
                5,
                (0, 0, 255),
                -1
            )

            # Desenhar linhas dos defects
            cv.line(
                result,
                start,
                far,
                (0, 255, 255),
                1
            )

            cv.line(
                result,
                end,
                far,
                (0, 255, 255),
                1
            )

    else:
        print("Nenhum Convexity Defect encontrado.")

    # 7. Mostrar os resultados
    cv.imshow("Imagem Original", image)
    cv.imshow("Mascara Binaria", mask)
    cv.imshow("Hand Geometry", result)

    # Esperar ate o utilizador pressionar uma tecla
    cv.waitKey(0)

    # Fechar todas as janelas
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
