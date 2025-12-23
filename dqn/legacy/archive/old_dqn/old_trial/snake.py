import pygame
import random
import sys

# ---------- CONFIG ----------
CELL_SIZE = 50          # pixels per grid cell
GRID_WIDTH = 10         # number of cells horizontally
GRID_HEIGHT = 10        # number of cells vertically
WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT
FPS = 10                # game speed

# Colors (R, G, B)
BLACK = (0, 0, 0)
DARK_GRAY = (40, 40, 40)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
WHITE = (255, 255, 255)


def random_empty_cell(snake):
    """Return a random grid position not occupied by the snake."""
    while True:
        x = random.randint(0, GRID_WIDTH - 1)
        y = random.randint(0, GRID_HEIGHT - 1)
        if (x, y) not in snake:
            return (x, y)


def draw_cell(surface, pos, color):
    """Draw a single cell at grid position pos = (x, y)."""
    x, y = pos
    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, color, rect)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 36)

    # Initial snake in center, length 3
    start_x = GRID_WIDTH // 2
    start_y = GRID_HEIGHT // 2
    snake = [
        (start_x, start_y),
        (start_x - 1, start_y),
        (start_x - 2, start_y),
    ]
    direction = (1, 0)  # moving right: (dx, dy)

    food = random_empty_cell(snake)
    score = 0
    running = True
    game_over = False

    while running:
        # ----- EVENT HANDLING -----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if not game_over:
                    # Change direction (no reversing allowed)
                    if event.key == pygame.K_UP and direction != (0, 1):
                        direction = (0, -1)
                    elif event.key == pygame.K_DOWN and direction != (0, -1):
                        direction = (0, 1)
                    elif event.key == pygame.K_LEFT and direction != (1, 0):
                        direction = (-1, 0)
                    elif event.key == pygame.K_RIGHT and direction != (-1, 0):
                        direction = (1, 0)
                else:
                    # On game over, press SPACE to restart
                    if event.key == pygame.K_SPACE:
                        return main()

        if not game_over:
            # ----- UPDATE GAME STATE -----
            head_x, head_y = snake[0]
            dx, dy = direction
            new_head = (head_x + dx, head_y + dy)

            # Check collisions with walls
            x, y = new_head
            if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                game_over = True
            # Check collision with self
            elif new_head in snake:
                game_over = True
            else:
                # Move snake
                snake.insert(0, new_head)  # add new head

                if new_head == food:
                    # Eat food: increase score and spawn new food
                    score += 1
                    food = random_empty_cell(snake)
                    # Don't pop tail -> snake grows
                else:
                    # Normal move: remove tail
                    snake.pop()

        # ----- DRAW -----
        screen.fill(BLACK)

        # Optional: grid lines
        for x in range(GRID_WIDTH):
            pygame.draw.line(
                screen, DARK_GRAY,
                (x * CELL_SIZE, 0),
                (x * CELL_SIZE, WINDOW_HEIGHT),
                1,
            )
        for y in range(GRID_HEIGHT):
            pygame.draw.line(
                screen, DARK_GRAY,
                (0, y * CELL_SIZE),
                (WINDOW_WIDTH, y * CELL_SIZE),
                1,
            )

        # Draw food
        draw_cell(screen, food, RED)

        # Draw snake
        for i, segment in enumerate(snake):
            color = GREEN
            draw_cell(screen, segment, color)

        # Draw score
        score_surf = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_surf, (10, 10))

        if game_over:
            over_text = font.render("Game Over! Press SPACE", True, WHITE)
            rect = over_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            screen.blit(over_text, rect)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
