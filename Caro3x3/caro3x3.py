'''Nhoms 22
N20DCCN053-Nguyễn Đình Phát
N20DCCN055-Lê Văn Phúc
N20DCCN083-Nguyễn Thái Trưởng
'''
import pygame
import time
import sys
#Mảng quản lí các ô cờ
grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
#Giới hạn các giá trị max min
MAX = 100
MIN = -100
#Mã màu
BLACK = (0, 0, 0)
WRITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 102)
BLUESKY = (0, 204, 204)
#Frame per second
FPS = 120
#Quy định X = 1, O = -1
checkX = 1
checkO = -1
#Kích thước 1 ô trên bàn cờ
WIDTH = 28
HEITH = 28
#Khoảng cách giữa các ô
MARGIN = 2
ROWNUM = 33
COLNUM = 64
def isMovesLeft(grid):
    for i in range(3):
        for j in range(3):
            if (grid[i][j] == 0):
                return True
    return False
def evaluate(grid):
    # Kiểm tra X hoặc O thắng theo hàng
    for row in range(3):
        if (grid[row][0] == grid[row][1] and grid[row][1] == grid[row][2]):
            if (grid[row][0] == checkX):
                return 10
            elif (grid[row][0] == checkO):
                return -10

    # Kiểm tra X hoặc O thắng theo cột
    for col in range(3):

        if (grid[0][col] == grid[1][col] and grid[1][col] == grid[2][col]):

            if (grid[0][col] == checkX):
                return 10
            elif (grid[0][col] == checkO):
                return -10

    # Kiểm tra X hoặc O thắng theo đường chéo chính
    if (grid[0][0] == grid[1][1] and grid[1][1] == grid[2][2]):

        if (grid[0][0] == checkX):
            return 10
        elif (grid[0][0] == checkO):
            return -10
    #Kiểm tra theo đường chéo phụ
    if (grid[0][2] == grid[1][1] and grid[1][1] == grid[2][0]):

        if (grid[0][2] == checkX):
            return 10
        elif (grid[0][2] == checkO):
            return -10

    # Chưa thắng = 0
    return 0
def alphabeta(board, depth, a, b, isMax):
    #Tính điểm trên bàn cờ
    score = evaluate(board)
    #Có điểm thì kết thúc bàn cờ
    if score != 0:
        return score

    # Hết ô để đánh mà vẫn chưa có điểm nào = hòa
    if (isMovesLeft(board) == False):
        return 0

    # Max đi
    if (isMax):
        best = MIN
        # Kiểm tra các ô
        for i in range(3):
            for j in range(3):
                #Nếu ô chưa đánh
                if (board[i][j] == 0):
                    board[i][j] = checkX
                    best = max(best, alphabeta(board, depth + 1, a, b, not isMax))
                    a = max(best, a)
                    #Hồi lại nước đã đi
                    board[i][j] = 0
                    if b <= a:
                        break
        return best
    #Min đi
    else:
        best = MAX
        for i in range(3):
            for j in range(3):
                if (board[i][j] == 0):
                    board[i][j] = checkO
                    best = min(best, alphabeta(board, depth + 1, a, b, not isMax))
                    b = min(best, b)
                    # Hồi lại nước đã đi
                    board[i][j] = 0
                    if b <= a:
                        break
        return best
def findBestMove(board):
    bestVal = 2
    bestMove = (-1, -1)
    for i in range(3):
        for j in range(3):
            if (board[i][j] == 0):
                board[i][j] = checkO
                moveVal = alphabeta(board, 0, -2, 2, True)
                # Hồi lại bước đi
                board[i][j] = 0
                #Nếu nước đi hiện tại tốt hơn bestVal thì cập nhật bestVal
                if (moveVal < bestVal):
                    bestMove = (i, j)
                    bestVal = moveVal
    return bestMove
WINDOW_SIZE = (640, 430)
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.flip()
pygame.display.set_caption("CARO 3x3")
# pygame.display.set_icon()
x_img = pygame.transform.smoothscale(pygame.image.load("tải xuống.png").convert(),(80,122))
o_img = pygame.transform.smoothscale(pygame.image.load("images.png").convert(),(80,122))
running = True
clock = pygame.time.Clock()
#Tọa đồ bắt đầu vị trí của bàn cờ
x = 200
y = 25
turn = 0
while running:
    result = evaluate(grid)
    screen.fill(BLUESKY)
    cellIndexX = 0
    for i in range(3):
        for j in range(3):
            pygame.draw.rect(screen, WRITE, pygame.Rect(x + i*80, y+j*122, 80, 122), 1)
            if grid[i][j] == 1:
                screen.blit(x_img, (x + j*80, y + i*122))
                # pygame.draw.rect(screen, RED, pygame.Rect(x + j*80, y + i*122, 78, 120))
            if grid[i][j] == -1:
                screen.blit(o_img, (x + j * 80, y + i * 122))
                # pygame.draw.rect(screen, GREEN, pygame.Rect(x + j * 80, y + i * 122, 78, 120))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            exit();
        if turn%2 == 0:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                col = pos[0]
                row = pos[1]
                if col > 200 and col < 280:
                    if row > 25 and row < 147:
                        if grid[0][0] == 0:
                            grid[0][0] = 1
                            turn += 1
                    if row > 147 and row < 269:
                        if grid[1][0] == 0:
                            grid[1][0] = 1
                            turn += 1
                    if row > 269 and row < 391:
                        if grid[2][0] == 0:
                            grid[2][0] = 1
                            turn += 1
                elif col > 280 and col < 360:
                    if row > 25 and row < 147:
                        if grid[0][1] == 0:
                            grid[0][1] = 1
                            turn += 1
                    if row > 147 and row < 269:
                        if grid[1][1] == 0:
                            grid[1][1] = 1
                            turn += 1
                    if row > 269 and row < 391:
                        if grid[2][1] == 0:
                            grid[2][1] = 1
                            turn += 1
                elif col > 360 and col < 440:
                    if row > 25 and row < 147:
                        if grid[0][2] == 0:
                            grid[0][2] = 1
                            turn += 1
                    if row > 147 and row < 269:
                        if grid[1][2] == 0:
                            grid[1][2] = 1
                            turn += 1
                    if row > 269 and row < 391:
                        if grid[2][2] == 0:
                            grid[2][2] = 1
                            turn += 1
        else:
            botMove = findBestMove(grid)
            grid[botMove[0]][botMove[1]] = -1
            turn += 1
    pygame.display.update()
    clock.tick(FPS)
    if result != 0:
        running = False
        time.sleep(1)
    if turn > 9:
        running = False
        time.sleep(1)
if result == 10:
    print("X thắng")
elif result == -10:
    print("O thắng")
else:
    print("Hòa")

