"""
Drawing text with pyglet label
"""
import math
from typing import Tuple, Union

import arcade
import pyglet
from arcade.arcade_types import Color, Point
from arcade.draw_commands import get_four_byte_color
from pyglet.math import Mat4
from arcade.resources import resolve_resource_path




def load_font(font_name) -> None:
    """
    Load a font for later use.


    :param font_name:
    :raises FileNotFoundError: if the font specified wasn't found
    :return:
    """
    # search resources folder for the named font, and error if it doesn't exist
    if font_name.startswith(":resources:"):
        try:
            file_path = resolve_resource_path(font_name)
        except FileNotFoundError:
            raise FileNotFoundError(f"Unable to find resource with the name: {font_name}")
    else:
        file_path = font_name

    pyglet.font.add_file(str(file_path))



FontNameOrNames = Union[str, Tuple[str, ...]]


def _attempt_font_name_resolution(font_name: FontNameOrNames) -> FontNameOrNames:
    """
    Attempt to resolve a tuple of font names.

    Preserves the original logic of this section, even though it
    doesn't seem to make sense entirely. Comments are an attempt
    to make sense of the original code.

    If it can't resolve a definite path, it will return the original
    argument for pyglet to attempt to resolve. This is consistent with
    the original behavior of this code before it was encapsulated.

    :param Union[str, Tuple[str, ...]] font_name:
    :return: Either a resolved path or the original tuple
    """
    if font_name:

        # ensure
        if isinstance(font_name, str):
            font_list: Tuple[str, ...] = (font_name,)
        elif isinstance(font_name, tuple):
            font_list = font_name
        else:
            raise TypeError("font_name parameter must be a string, or a tuple of strings that specify a font name.")

        for font in font_list:
            try:
                path = resolve_resource_path(font)
                # print(f"Font path: {path=}")

                # found a font successfully!
                return path.name

            except FileNotFoundError:
                pass

    # failed to find it ourselves, hope pyglet can make sense of it
    return font_name


def _draw_label_with_rotation(label: pyglet.text.Label, rotation: float) -> None:
    """

    Helper for drawing pyglet labels with rotation within arcade.

    Originally part of draw_text in this module, now abstracted and improved
    so that both arcade.Text and arcade.draw_text can make use of it.

    :param pyglet.text.Label label: a pyglet label to wrap and draw
    :param float rotation: rotate this many degrees from horizontal around anchor
    """

    # raw pyglet draw functions need this context helper inside arcade
    window = arcade.get_window()
    with window.ctx.pyglet_rendering():

        # execute view matrix magic to rotate cleanly
        if rotation:
            # original_view = window.view

            angle_radians = math.radians(rotation)
            x = label.x
            y = label.y
            label.x = 0
            label.y = 0
            r_view = Mat4.from_rotation(angle_radians, (0, 0, 1))
            t_view = Mat4.from_translation((x, y, 0))
            final_view = r_view @ t_view
            window.view = final_view

        label.draw()

        # restore original position if we used view matrix magic
        if rotation:
            # linters might warn that this is used before assignment,
            # but it's actually valid since we only use it when it was
            # previously assigned.
            label.x, label.y = x, y



class Text:
    """
    An object-oriented way to draw text to the screen.

    .. tip:: Use this class when performance matters!

       Unlike :py:func:`~arcade.draw_text`, this class does not risk
       wasting time recalculating and re-setting any text each time
       :py:meth:`~arcade.Text.draw` is called. This makes it faster
       while:

       - requiring you to manage instances and drawing yourself
       - using negligible extra RAM

       The speed advantage scales as more text needs to be drawn
       to the screen.

    The constructor arguments work identically to those of
    :py:func:`~arcade.draw_text`. See its documentation for in-depth
    explanation for how to use each of them. For example code, see :ref:`drawing_text_objects`.

    :param str text: Initial text to display. Can be an empty string
    :param float start_x: x position to align the text's anchor point with
    :param float start_y: y position to align the text's anchor point with
    :param Color color: Color of the text as a tuple or list of 3 (RGB) or 4 (RGBA) integers
    :param float font_size: Size of the text in points
    :param float width: A width limit in pixels
    :param str align: Horizontal alignment; values other than "left" require width to be set
    :param Union[str, Tuple[str, ...]] font_name: A font name, path to a font file, or list of names
    :param bool bold: Whether to draw the text as bold
    :param bool italic: Whether to draw the text as italic
    :param str anchor_x: How to calculate the anchor point's x coordinate
    :param str anchor_y: How to calculate the anchor point's y coordinate
    :param bool multiline: Requires width to be set; enables word wrap rather than clipping
    :param float rotation: rotation in degrees, counter-clockwise from horizontal

    All constructor arguments other than ``text`` have a corresponding
    property. To access the current text, use the ``value`` property
    instead.

    By default, the text is placed so that:

    - the left edge of its bounding box is at ``start_x``
    - its baseline is at ``start_y``

    The baseline is located along the line the bottom of the text would
    be written on, excluding letters with tails such as y:

        .. figure:: ../images/text_anchor_y.png
           :width: 40%

           The blue line is the baseline for the string ``"Python"``

    ``rotation`` allows for the text to be rotated around the anchor
    point by the passed number of degrees. Positive values rotate
    counter-clockwise from horizontal, while negative values rotate
    clockwise:

        .. figure:: ../images/text_rotation_degrees.png
           :width: 55%

           Rotation around the default anchor (
           ``anchor_y="baseline"`` and ``anchor_x="left"``)

    """

    def __init__(
        self,
        text: str,
        start_x: float,
        start_y: float,
        color: Color = arcade.color.WHITE,
        font_size: float = 12,
        width: int = 0,
        align: str = "left",
        font_name: FontNameOrNames = ("calibri", "arial"),
        bold: bool = False,
        italic: bool = False,
        anchor_x: str = "left",
        anchor_y: str = "baseline",
        multiline: bool = False,
        rotation: float = 0
    ):
        """Build a text object"""

        if align != "center" and align != "left" and align != "right":
            raise ValueError("The 'align' parameter must be equal to 'left', 'right', or 'center'.")

        if align != "left":
            multiline = True

        adjusted_font = _attempt_font_name_resolution(font_name)
        self._label = pyglet.text.Label(
            text=text,
            x=start_x,
            y=start_y,
            font_name=adjusted_font,
            font_size=font_size,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            color=get_four_byte_color(color),
            width=width,
            align=align,
            bold=bold,
            italic=italic,
            multiline=multiline
        )
        self.rotation = rotation

    @property
    def value(self) -> str:
        """
        The current value to display.
        """
        return self._label.text

    @value.setter
    def value(self, value: str):
        self._label.text = value

    @property
    def x(self) -> float:
        return self._label.x

    @x.setter
    def x(self, x: float) -> None:
        self._label.x = x

    @property
    def y(self) -> float:
        return self._label.y

    @y.setter
    def y(self, y: float):
        self._label.y = y

    @property
    def font_name(self) -> FontNameOrNames:
        return self._label.font_name

    @font_name.setter
    def font_name(self, font_name: FontNameOrNames) -> None:
        self._label.font_name = font_name

    @property
    def font_size(self) -> float:
        return self._label.font_size

    @font_size.setter
    def font_size(self, font_size: float):
        self._label.font_size = font_size

    @property
    def anchor_x(self) -> str:
        return self._label.anchor_x

    @anchor_x.setter
    def anchor_x(self, anchor_x: str):
        self._label.anchor_x = anchor_x

    @property
    def anchor_y(self) -> str:
        return self._label.anchor_y

    @anchor_y.setter
    def anchor_y(self, anchor_y: str):
        self._label.anchor_y = anchor_y

    @property
    def color(self) -> Color:
        return self._label.color

    @color.setter
    def color(self, color: Color):
        self._label.color = get_four_byte_color(color)

    @property
    def width(self) -> int:
        return self._label.width

    @width.setter
    def width(self, width: int):
        self._label.width = width

    @property
    def align(self) -> str:
        return self._label.get_style("align")  # type: ignore

    @align.setter
    def align(self, align: str):

        # duplicates the logic used in the rest of this module
        if align != "left":
            self.multiline = True

        self._label.set_style("align", align)

    @property
    def bold(self) -> bool:
        return self._label.bold

    @bold.setter
    def bold(self, bold: bool):
        self._label.bold = bold

    @property
    def italic(self) -> bool:
        return self._label.italic

    @italic.setter
    def italic(self, italic: bool):
        self._label.italic = italic

    @property
    def multiline(self) -> bool:
        return self._label.multiline

    @multiline.setter
    def multiline(self, multiline: bool):
        self._label.multiline = multiline

    def draw(self) -> None:
        """
        Draw this label to the screen at its current ``x`` and ``y`` position.

        .. warning: Cameras affect text drawing!
            If you want to draw a custom GUI that doesn't move with the
            game world, you will need a second :py:class:`~arcade.Camera`
            instance. For information on how to do this, see
            :ref:`sprite_move_scrolling`.

        """
        _draw_label_with_rotation(self._label, self.rotation)


    @property
    def position(self) -> Point:
        """
        The current x, y position as a tuple. This wraps x and y.
        """
        return self._label.x, self._label.y

    @position.setter
    def position(self, point: Point):
        self._label.x, self._label.y = point





def draw_text(
    text: str,
    start_x: float,
    start_y: float,
    color: Color = arcade.color.WHITE,
    font_size: float = 12,
    width: int = 0,
    align: str = "left",
    font_name: FontNameOrNames = ("calibri", "arial"),
    bold: bool = False,
    italic: bool = False,
    anchor_x: str = "left",
    anchor_y: str = "baseline",
    multiline: bool = False,
    rotation: float = 0,
):
    """
    A simple way for beginners to draw text.

    .. warning:: Cameras affect text drawing!

        If you want to draw a custom GUI that doesn't move with the
        game world, you will need a second camera. For information on
        how to do this, see :ref:`sprite_move_scrolling`.

    This function lets you start draw text easily with better
    performance than the old pillow-based text. If you need even higher
    performance, consider using :py:class:`~arcade.Text`.

    Example code can be found at :ref:`drawing_text`.

    :param str text: Text to display
    :param float start_x: x position to align the text's anchor point with
    :param float start_y: y position to align the text's anchor point with
    :param Color color: Color of the text as a tuple or list of 3 (RGB) or 4 (RGBA) integers
    :param float font_size: Size of the text in points
    :param float width: A width limit in pixels
    :param str align: Horizontal alignment; values other than "left" require width to be set
    :param Union[str, Tuple[str, ...]] font_name: A font name, path to a font file, or list of names
    :param bool bold: Whether to draw the text as bold
    :param bool italic: Whether to draw the text as italic
    :param str anchor_x: How to calculate the anchor point's x coordinate
    :param str anchor_y: How to calculate the anchor point's y coordinate
    :param bool multiline: Requires width to be set; enables word wrap rather than clipping
    :param float rotation: rotation in degrees, counter-clockwise from horizontal

    By default, the text is placed so that:

    - the left edge of its bounding box is at ``start_x``
    - its baseline is at ``start_y``

    The baseline of text is the line it would be written on:

        .. figure:: ../images/text_anchor_y.png
           :width: 40%

           The blue line is the baseline for the string ``"Python"``

    ``font_name`` can be any of the following:

    - a built-in font in the :ref:`Resources`
    - the name of a system font
    - a path to a font on the system
    - a `tuple` containing any mix of the previous three

    Each entry provided will be tried in order until one is found. If
    none of the fonts are found, a default font will be chosen (usually
    Arial).

    ``anchor_x`` and ``anchor_y`` specify how to calculate the anchor point,
    which affects how the text is:

    - Placed relative to ``start_x`` and ``start_y``
    - Rotated

    By default, the text is drawn so that ``start_x`` is at the left of
    the text's bounding box and ``start_y`` is at the baseline.

    You can set a custom anchor point by passing combinations of the
    following values for ``anchor_x`` and ``anchor_y``:

    .. list-table:: Values allowed by ``anchor_x``
        :widths: 20 40 40
        :header-rows: 1

        * - String value
          - Practical Effect
          - Anchor Position

        * - ``"left"`` `(default)`
          - Text drawn with its left side at ``start_x``
          - Anchor point at the left side of the text's bounding box

        * - ``"center"``
          - Text drawn horizontally centered on ``start_x``
          - Anchor point at horizontal center of text's bounding box

        * - ``"right"``
          - Text drawn with its right side at ``start_x``
          - Anchor placed at the right side of the text's bounding box


    .. list-table:: Values allowed by ``anchor_y``
        :widths: 20 40 40
        :header-rows: 1

        * - String value
          - Practical Effect
          - Anchor Position

        * - ``"baseline"`` `(default)`
          - Text drawn with baseline on ``start_y``.
          - Anchor placed at the text rendering baseline

        * - ``"top"``
          - Text drawn with its top aligned with ``start_y``
          - Anchor point placed at the top of the text

        * - ``"bottom"``
          - Text drawn with its absolute bottom aligned with ``start_y``,
            including the space for tails on letters such as y and g
          - Anchor point placed at the bottom of the text after the
            space allotted for letters such as y and g

        * - ``"center"``
          - Text drawn with its vertical center on ``start_y``
          - Anchor placed at the vertical center of the text


    ``rotation`` allows for the text to be rotated around the anchor
    point by the passed number of degrees. Positive values rotate
    counter-clockwise from horizontal, while negative values rotate
    clockwise:

        .. figure:: ../images/text_rotation_degrees.png
           :width: 55%

           Rotation around the default anchor point (
           ``anchor_y="baseline"`` and ``anchor_x="left"``)


    It can be helpful to think of this function working as follows:

    1. Text layout and alignment are calculated:

        1. The text's characters are laid out within a bounding box
           according to the current styling options

        2. The anchor point on the text is calculated based on
           the text value, styling, as well as values for ``anchor_x``
           and ``anchor_y``

    2. The text is placed so its anchor point is at ``(start_x,
       start_y))``

    3. The text is rotated around its anchor point before finally
       being drawn

    This function is less efficient than using :py:class:`~arcade.Text`
    because some of the steps above can be repeated each time a call is
    made rather than fully cached as with the class.

    """
    # See : https://github.com/pyglet/pyglet/blob/ff30eadc2942553c9de96d6ce564ad1bc3128fb4/pyglet/text/__init__.py#L401

    color = get_four_byte_color(color)
    # Cache the states that are expensive to change
    key = f"{font_size}{font_name}{bold}{italic}{anchor_x}{anchor_y}{align}{width}"
    cache = arcade.get_window().ctx.pyglet_label_cache
    label = cache.get(key)
    if align != "center" and align != "left" and align != "right":
        raise ValueError("The 'align' parameter must be equal to 'left', 'right', or 'center'.")

    if align != "left":
        multiline = True

    if not label:
        adjusted_font = _attempt_font_name_resolution(font_name)

        label = pyglet.text.Label(
            text=text,
            x=start_x,
            y=start_y,
            font_name=adjusted_font,
            font_size=font_size,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            color=color,
            width=width,
            align=align,
            bold=bold,
            italic=italic,
            multiline=multiline,
        )
        cache[key] = label

    # These updates are relatively cheap
    label.text = text
    label.x = start_x
    label.y = start_y
    label.color = color

    _draw_label_with_rotation(label, rotation)



# TODO: maybe remove, as this is invalid

def create_text(*args, **kwargs):
    """Legacy stub, returns a text object."""
    return Text("Hello")
