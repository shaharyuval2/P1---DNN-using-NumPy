import numpy as np
import pygame
from scipy import signal

from p1_dnn.models import NeuralNetwork

# screen and board global variables
WIDTH, HEIGHT = 900, 700
PIXEL_MULTIPLE = 13
BOARD_SIZE = 28
BOARD_WIDTH = BOARD_SIZE * PIXEL_MULTIPLE
BOARD_HEIGHT = BOARD_SIZE * PIXEL_MULTIPLE
BOARD = None

COLOURS = {
    "main": (54, 56, 47),
    "secondary": (105, 117, 101),
    "green": (85, 110, 83),
    "third": (255, 242, 223),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}

# IMG, Kernel, Guess global variables
IMG = np.zeros((BOARD_SIZE, BOARD_SIZE))
IMG_SMOOTHED = np.zeros((BOARD_SIZE, BOARD_SIZE))
K_CORNER = 0.1
K_SIDES = 0.4
KERNEL = np.array(
    [
        [K_CORNER, K_SIDES, K_CORNER],
        [K_SIDES, 1, K_SIDES],
        [K_CORNER, K_SIDES, K_CORNER],
    ]
)
GUESS = None
CONFIDENCE = None

# load model
try:
    nn = NeuralNetwork.load("models/epoch100_batchsize100_eta0.01_feb27_16:18.npz")
except:
    print("Model could not load, try different file name")
    exit()


pygame.init()
screen = pygame.display.set_mode([WIDTH, HEIGHT])

fps = 60
clock = pygame.time.Clock()

button_font = pygame.font.SysFont("verdana", 34)
guess_font = pygame.font.SysFont("Verdana", 22)
pygame.display.set_caption("My Neural Net Showcase!")


class Button:
    def __init__(self, color, x, y, width, height, radius, text=""):
        self.color = color
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.radius = radius

    def draw(self, screen):
        # shadow
        pygame.draw.rect(
            screen,
            (0.2 * self.color[0], 0.2 * self.color[1], 0.2 * self.color[2]),
            [self.rect.x + 5, self.rect.y + 5, self.rect.width, self.rect.height],
            border_radius=self.radius,
        )
        # button
        pygame.draw.rect(screen, self.color, self.rect, border_radius=self.radius)
        # button text
        if self.text:
            text_surf = button_font.render(self.text, True, COLOURS["third"])
            text_rect = text_surf.get_rect(center=self.rect.center)
            screen.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


# ----- Helper Functions -----
def lerp_color(colorA, colorB, p):
    return (
        colorA[0] * (1 - p) + colorB[0] * p,
        colorA[1] * (1 - p) + colorB[1] * p,
        colorA[2] * (1 - p) + colorB[2] * p,
    )


def shift_rect(rect, shift_vec):
    return [rect[0] + shift_vec[0], rect[1] + shift_vec[1], rect[2], rect[3]]


def pad_rect(rect, padding):
    return [
        rect[0] - padding,
        rect[1] - padding,
        rect[2] + 2 * padding,
        rect[3] + 2 * padding,
    ]


def pad_rect_non_symmetrical(rect, padding_x, padding_y):
    return [
        rect[0] - padding_x,
        rect[1] - padding_y,
        rect[2] + 2 * padding_x,
        rect[3] + 2 * padding_y,
    ]


# ----- Draw Helper Functions -----


def draw_text(text, font, color, text_center):
    text_surf = font.render(
        text,
        True,
        color,
    )
    text_rect = text_surf.get_rect(center=text_center)
    screen.blit(text_surf, text_rect)


def draw_rect(func_screen, color, rect, border_radius=0, border=False, shadow=False):
    # draw shadow
    if shadow:
        pygame.draw.rect(
            func_screen,
            lerp_color(color, (0, 0, 0), 0.8),
            shift_rect(rect, (10, 10)),
            border_radius=border_radius,
        )
    # draw rect
    rect = pygame.draw.rect(func_screen, color, rect, border_radius=border_radius)
    # draw border
    if border:
        pygame.draw.rect(
            func_screen,
            COLOURS["third"],
            pad_rect(rect, 2),
            2,
            border_radius=border_radius,
        )
    return rect


# ----- Draw Screen Components Functions -----
def draw_board():
    board = pygame.draw.rect(
        screen,
        "black",
        [
            (WIDTH - BOARD_WIDTH) / 2 - 150,
            (HEIGHT - BOARD_HEIGHT) / 2 - 100,
            BOARD_WIDTH,
            BOARD_HEIGHT,
        ],
    )

    pygame.draw.rect(
        screen,
        COLOURS["third"],
        [
            board.x - 5,
            board.y - 5,
            BOARD_WIDTH + 10,
            BOARD_HEIGHT + 10,
        ],
        5,
        border_radius=2,
    )

    return board


def paint_board():
    gray_values = (IMG_SMOOTHED * 255).astype(np.uint8)
    # rgb_IMG[height][width] = [gray_value, gray_value, gray_value]
    rgb_IMG = np.stack((gray_values,) * 3, axis=-1)
    # rgb_IMG[width][height] = [gray_value, gray_value, gray_value]
    rgb_IMG = np.transpose(rgb_IMG, (1, 0, 2))

    temp_surf = pygame.surfarray.make_surface(rgb_IMG)
    final_size = (BOARD_WIDTH, BOARD_HEIGHT)
    scaled_surf = pygame.transform.scale(temp_surf, final_size)
    screen.blit(scaled_surf, BOARD.topleft)


def draw_painting_area():
    pygame.draw.rect(
        screen,
        "red",
        [
            BOARD.topleft[0] + 4 * PIXEL_MULTIPLE - 2,
            BOARD.topleft[1] + 4 * PIXEL_MULTIPLE - 2,
            20 * PIXEL_MULTIPLE + 4,
            20 * PIXEL_MULTIPLE + 4,
        ],
        2,
    )

    painting_area = pygame.Rect(
        BOARD.topleft[0] + 5 * PIXEL_MULTIPLE,
        BOARD.topleft[1] + 5 * PIXEL_MULTIPLE,
        18 * PIXEL_MULTIPLE,
        18 * PIXEL_MULTIPLE,
    )

    return painting_area


def draw_probs(probs):
    board = draw_rect(
        screen,
        COLOURS["secondary"],
        [BOARD.topright[0] + 100, BOARD.topright[1], 240, BOARD.height],
        border_radius=10,
        border=True,
        shadow=True,
    )

    draw_text(
        "Probabilities", button_font, COLOURS["third"], (board.centerx, board.top + 30)
    )
    for i in range(10):
        draw_text(
            f"{i}: {probs[i].item() * 100.0:.2f}%",
            guess_font,
            lerp_color(COLOURS["secondary"], COLOURS["third"], probs[i].item()),
            (board.centerx, board.top + i * 30 + 70),
        )


def draw_guess(guess):
    text_surf = guess_font.render(
        f"Machine Guess: {guess}",
        True,
        COLOURS["third"],
    )
    text_rect = text_surf.get_rect(center=(WIDTH / 2, BOARD.center[1] + 350))

    draw_rect(
        screen,
        COLOURS["secondary"],
        pad_rect_non_symmetrical(text_rect, 50, 25),
        10,
        border=True,
        shadow=True,
    )

    screen.blit(text_surf, text_rect)


# ----- IMG related Functions -----
def smooth_IMG(padding=1):
    global IMG_SMOOTHED
    IMG_SMOOTHED = signal.convolve2d(IMG, KERNEL, mode="same")
    np.clip(IMG_SMOOTHED, 0, 1, out=IMG_SMOOTHED)


def center_image(img):
    # Get the coordinates of all non-zero pixels
    rows, cols = np.where(img > 0)
    if len(rows) == 0:
        return img  # Image is empty

    # Calculate center of mass
    weights = img[rows, cols]
    m_y = np.average(rows, weights=weights)
    m_x = np.average(cols, weights=weights)

    # Calculate the shift required to get to (14, 14)
    shift_y = int(round(14 - m_y))
    shift_x = int(round(14 - m_x))

    # Use np.roll to shift the image
    centered_img = np.roll(img, shift_y, axis=0)
    centered_img = np.roll(centered_img, shift_x, axis=1)

    # Clean up "wrap-around" edges
    # np.roll wraps pixels from the right side to the left.
    # For a centered digit, we want those gaps to be 0 (black).
    if shift_y > 0:
        centered_img[:shift_y, :] = 0
    elif shift_y < 0:
        centered_img[shift_y:, :] = 0

    if shift_x > 0:
        centered_img[:, :shift_x] = 0
    elif shift_x < 0:
        centered_img[:, shift_x:] = 0

    return centered_img


def update_IMG():
    mouse_buttons = np.array(pygame.mouse.get_pressed())
    mouse_pos = np.array(pygame.mouse.get_pos())
    if mouse_buttons[0] and painting_area.collidepoint(mouse_pos):
        mouse_board_pos = mouse_pos - np.array(BOARD.topleft)
        pos_idx = mouse_board_pos // PIXEL_MULTIPLE
        IMG[pos_idx[1], pos_idx[0]] = 1

        # update IMG_SMOOTHED and GUESS
        smooth_IMG()
        guess, probs = machine_guess()
        draw_guess(guess)
        draw_probs(probs)


# ----- Pass IMG in the Model Function-----
def machine_guess():
    img_centered = center_image(IMG_SMOOTHED)
    img_column = img_centered.reshape(-1, 1)
    probs = nn.forward(img_column)
    guess = np.argmax(probs)

    return guess, probs


# draw static screen components
screen.fill(COLOURS["main"])
BOARD = draw_board()
clear_button = Button(
    COLOURS["secondary"],
    BOARD.bottomleft[0] + BOARD.width / 2 - 80,
    BOARD.bottomleft[1] + 30,
    160,
    60,
    20,
    "Clear",
)
clear_button.draw(screen)

# run initial forwardpass
guess, probs = machine_guess()
draw_guess(guess)
draw_probs(probs)

# ----- Main Loop -----
run = True
while run:
    clock.tick(fps)

    # paint drawing and painting area on board
    paint_board()
    painting_area = draw_painting_area()

    # update IMG and update probs and guess screen components accordingly
    update_IMG()

    # handle events
    for event in pygame.event.get():
        # window closed event
        if event.type == pygame.QUIT:
            run = False
        # 'clear' button pressed
        if event.type == pygame.MOUSEBUTTONDOWN:
            if clear_button.is_clicked(event.pos):
                IMG = np.zeros((BOARD_SIZE, BOARD_SIZE))
                smooth_IMG()
                guess, probs = machine_guess()
                draw_guess(guess)
                draw_probs(probs)

    pygame.display.flip()

pygame.quit()
