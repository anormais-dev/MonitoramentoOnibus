import cv2
import os
from salvar.classificador import classificar

# Pasta com imagens de teste
pastaTestes = "prova"
imagens = [f for f in os.listdir(pastaTestes) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for imagem in imagens:
    path = os.path.join(pastaTestes, imagem)
    img = cv2.imread(path)

    if img is None:
        print(f"❌ Não foi possível carregar {imagem}")
        continue

    tipo = classificar(img)
    print(f"{imagem} → {tipo}")

    # Exibir imagem com legenda
    cv2.putText(img, f"{tipo.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0) if tipo == "municipal" else (0, 255, 255), 2)
    cv2.imshow("Classificação", img)
    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()
