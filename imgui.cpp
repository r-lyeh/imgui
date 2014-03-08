//
// Copyright (c) 2009-2010 Mikko Mononen memon@inside.org
// Copyright (c) 2011-2014 Mario 'rlyeh' Rodriguez
// Copyright (c) 2013 Florian Deconinck
// Copyright (c) 2013 Adrien Herubel
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

// Source altered and distributed from https://github.com/r-lyeh/imgui

#include <stdio.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <vector>
#include "imgui.hpp"

#ifdef _MSC_VER
#   pragma warning(push)
#   pragma warning(disable: 4996) // _CRT_SECURE_NO_WARNINGS
#   define snprintf _snprintf
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ref: Laird Shaw, http://creativeandcritical.net/str-replace-c/

#include <string.h>
#include <stdlib.h>
#include <stddef.h>

char *replace_str(const char *str, const char *old, const char *recent)
{
    char *ret, *r;
    const char *p, *q;
    size_t oldlen = strlen(old);
    size_t count, retlen, newlen = strlen(recent);

    if (oldlen != newlen) {
        for (count = 0, p = str; (q = strstr(p, old)) != NULL; p = q + oldlen)
            count++;
        /* this is undefined if p - str > PTRDIFF_MAX */
        retlen = p - str + strlen(p) + count * (newlen - oldlen);
    } else
        retlen = strlen(str);

    if ((ret = (char*) malloc(retlen + 1)) == NULL)
        return NULL;

    for (r = ret, p = str; (q = strstr(p, old)) != NULL; p = q + oldlen) {
        /* this is undefined if q - p > PTRDIFF_MAX */
        ptrdiff_t l = q - p;
        memcpy(r, p, l);
        r += l;
        memcpy(r, recent, newlen);
        r += newlen;
    }
    strcpy(r, p);

    return ret;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static std::vector< unsigned int > colors;

#define alpha(c,a)     (( colors[(c)] & 0x00ffffff ) | (a<<24) )

#define white_alpha(x) alpha( 2, (unsigned char)((x)*255.0/256.0) )
#define gray_alpha(x)  alpha( 1, (unsigned char)((x)*255.0/256.0) )
#define black_alpha(x) alpha( 0, (unsigned char)((x)*255.0/256.0) )
#define theme_alpha(x) alpha( colors.size() - 1, (unsigned char)((x)*255.0/256.0) )

static void imguiResetColors() {
    colors = {
        imguiRGBA(0,0,0),           //black
        imguiRGBA(128,128,128),     //gray
        imguiRGBA(255,255,255),     //white
        imguiRGBA(255,196,64)       //themed, queue #1
    };
}

void imguiPushColor( unsigned int c ) {
    colors.push_back( c );
}
void imguiPopColor() {
    colors.pop_back();
}

void imguiPushColorTheme() {
    colors.push_back( colors[3] );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static std::vector<bool> enables;

static void imguiResetEnables() {
    enables = {true};
}

void imguiPushEnable( int enable ) {
    enables.push_back( enables.back() && enable > 0 );
}
void imguiPopEnable() {
    enables.pop_back();
}

#define enabled ( enables.back() )

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static int frame = 0, caret = 0;
static void imguiResetCaret() {
    caret = ( ( ( ++frame %= 60 ) / (60/(3*2)) ) % 2 ); // 60hz/20 = ~3 per sec, then blink %2
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int (*imguiRenderCalcText)(const char* text) = 0;

static const unsigned TEXT_POOL_SIZE = 8000;
static char g_textPool[TEXT_POOL_SIZE];
static unsigned g_textPoolSize = 0;
static const char* allocText(const char* text)
{
        unsigned len = strlen(text)+1;
        if (g_textPoolSize + len >= TEXT_POOL_SIZE)
                return 0;
        char* dst = &g_textPool[g_textPoolSize];
        memcpy(dst, text, len);
        g_textPoolSize += len;
        return dst;
}

static const unsigned GFXCMD_QUEUE_SIZE = 5000;
static imguiGfxCmd g_gfxCmdQueue[GFXCMD_QUEUE_SIZE];
static unsigned g_gfxCmdQueueSize = 0;

static void resetGfxCmdQueue()
{
        g_gfxCmdQueueSize = 0;
        g_textPoolSize = 0;
}

static void addGfxCmdScissor(int x, int y, int w, int h)
{
        if (g_gfxCmdQueueSize >= GFXCMD_QUEUE_SIZE)
                return;
        imguiGfxCmd& cmd = g_gfxCmdQueue[g_gfxCmdQueueSize++];
        cmd.type = IMGUI_GFXCMD_SCISSOR;
        cmd.flags = x < 0 ? 0 : 1;      // on/off flag.
        cmd.col = imguiRGBA(0,0,0,0);
        cmd.rect.x = (short)x;
        cmd.rect.y = (short)y;
        cmd.rect.w = (short)w;
        cmd.rect.h = (short)h;
}

static imguiGfxRect getLastGfxCmdRect()
{
    for ( int i = g_gfxCmdQueueSize; i-- > 0; ) {
        imguiGfxCmd& cmd = g_gfxCmdQueue[i];
        if( cmd.type == IMGUI_GFXCMD_RECT )
            return cmd.rect;
    }
    return imguiGfxRect();
}

static void addGfxCmdRect(float x, float y, float w, float h, unsigned int color)
{
        if (g_gfxCmdQueueSize >= GFXCMD_QUEUE_SIZE)
                return;
        imguiGfxCmd& cmd = g_gfxCmdQueue[g_gfxCmdQueueSize++];
        cmd.type = IMGUI_GFXCMD_RECT;
        cmd.flags = 0;
        cmd.col = color;
        cmd.rect.x = (short)(x*8.0f);
        cmd.rect.y = (short)(y*8.0f);
        cmd.rect.w = (short)(w*8.0f);
        cmd.rect.h = (short)(h*8.0f);
        cmd.rect.r = 0;
}

static void addGfxCmdLine(float x0, float y0, float x1, float y1, float r, unsigned int color)
{
        if (g_gfxCmdQueueSize >= GFXCMD_QUEUE_SIZE)
                return;
        imguiGfxCmd& cmd = g_gfxCmdQueue[g_gfxCmdQueueSize++];
        cmd.type = IMGUI_GFXCMD_LINE;
        cmd.flags = 0;
        cmd.col = color;
        cmd.line.x0 = (short)(x0*8.0f);
        cmd.line.y0 = (short)(y0*8.0f);
        cmd.line.x1 = (short)(x1*8.0f);
        cmd.line.y1 = (short)(y1*8.0f);
        cmd.line.r = (short)(r*8.0f);
}

static void addGfxCmdRoundedRect(float x, float y, float w, float h, float r, unsigned int color)
{
        if (g_gfxCmdQueueSize >= GFXCMD_QUEUE_SIZE)
                return;
        imguiGfxCmd& cmd = g_gfxCmdQueue[g_gfxCmdQueueSize++];
        cmd.type = IMGUI_GFXCMD_RECT;
        cmd.flags = 0;
        cmd.col = color;
        cmd.rect.x = (short)(x*8.0f);
        cmd.rect.y = (short)(y*8.0f);
        cmd.rect.w = (short)(w*8.0f);
        cmd.rect.h = (short)(h*8.0f);
        cmd.rect.r = (short)(r*8.0f);
}

static void addGfxCmdTriangle(int x, int y, int w, int h, int flags, unsigned int color)
{
        if (g_gfxCmdQueueSize >= GFXCMD_QUEUE_SIZE)
                return;
        imguiGfxCmd& cmd = g_gfxCmdQueue[g_gfxCmdQueueSize++];
        cmd.type = IMGUI_GFXCMD_TRIANGLE;
        cmd.flags = (char)flags;
        cmd.col = color;
        cmd.rect.x = (short)(x*8.0f);
        cmd.rect.y = (short)(y*8.0f);
        cmd.rect.w = (short)(w*8.0f);
        cmd.rect.h = (short)(h*8.0f);
}

static void addGfxCmdText(int x, int y, int align, const char* text, unsigned int color)
{
        if (g_gfxCmdQueueSize >= GFXCMD_QUEUE_SIZE)
                return;
        imguiGfxCmd& cmd = g_gfxCmdQueue[g_gfxCmdQueueSize++];
        cmd.type = IMGUI_GFXCMD_TEXT;
        cmd.flags = 0;
        cmd.col = color;
        cmd.text.x = (short)x;
        cmd.text.y = (short)y;
        cmd.text.align = align;
        cmd.text.text = allocText(text);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct coord {
        int widgetX, widgetY, widgetW;
        coord() : widgetX(0),widgetY(0),widgetW(100)
        {}
};

#include <vector>

struct GuiState : public coord
{
        GuiState() :
                left(false), leftPressed(false), leftReleased(false),
                mx(-1), my(-1), scroll(0), ascii(0), lastAscii(0),
                inputable(0), active(0), hot(0), hotToBe(0), isHot(false), isActive(false), wentActive(false),
                dragX(0), dragY(0), dragOrig(0),
                insideCurrentScroll(false),  areaId(0), widgetId(0)
        {
        }

        bool left;
        bool leftPressed, leftReleased;
        int mx,my;
        int scroll;
        char ascii;
        char lastAscii;
        unsigned int inputable;
        unsigned int active;
        unsigned int hot;
        unsigned int hotToBe;
        bool isHot;
        bool isActive;
        bool wentActive;
        int dragX, dragY;
        float dragOrig;

        std::vector<coord> coords;
        void clear() {
            coords.clear();
        }
        void push() {
            coords.push_back(*this);
        }
        void set( int pos ) {
			if (pos < 0) pos = coords.size() - 1 + pos;
			*((coord*)this) = coords.at(pos);
        }

        bool insideCurrentScroll;

        unsigned int areaId;
        unsigned int widgetId;
};

static GuiState g_state;

void imguiStackPush() {
    g_state.push();
}
int imguiStackSet(int pos) {
    int cur = g_state.coords.size() - 1;
    g_state.set(pos);
    return cur;
}
void imguiSpaceDiv() {
    g_state.widgetW /= 2;
    g_state.push();
}
void imguiSpaceMul() {
    g_state.widgetW *= 2;
    g_state.push();
}
void imguiSpaceShift() {
    g_state.widgetX += g_state.widgetW;
    g_state.push();
}
void imguiSpaceUnshift() {
    g_state.widgetX -= g_state.widgetW;
    g_state.push();
}

inline bool anyActive()
{
        return g_state.active != 0;
}

inline bool isActive(unsigned int id)
{
        return g_state.active == id;
}

inline bool isInputable(unsigned int id)
{
        return g_state.inputable == id;
}

inline bool isHot(unsigned int id)
{
        return g_state.hot == id;
}

inline bool anyHot()
{
    return g_state.hot != 0;
}

inline bool inRect(int x, int y, int w, int h, bool checkScroll = true)
{
   return (!checkScroll || g_state.insideCurrentScroll) && g_state.mx >= x && g_state.mx <= x+w && g_state.my >= y && g_state.my <= y+h;
}

inline void clearInput()
{
        g_state.leftPressed = false;
        g_state.leftReleased = false;
        g_state.scroll = 0;
}

inline void clearActive()
{
        g_state.active = 0;
        // mark all UI for this frame as processed
        clearInput();
}

inline void setActive(unsigned int id)
{
        g_state.active = id;
        g_state.inputable = 0;
        g_state.wentActive = true;
}

inline void setInputable(unsigned int id){
    g_state.inputable = id;
}

inline void setHot(unsigned int id)
{
   g_state.hotToBe = id;
}


static bool buttonLogic(unsigned int id, bool over)
{
        bool res = false;
        // process down
        if (!anyActive())
        {
                if (over)
                        setHot(id);
                if (isHot(id) && g_state.leftPressed)
                        setActive(id);
        }

        // if button is active, then react on left up
        if (isActive(id))
        {
                g_state.isActive = true;
                if (over)
                        setHot(id);
                if (g_state.leftReleased)
                {
                        if (isHot(id))
                                res = true;
                        clearActive();
                }
        }

        if (isHot(id))
                g_state.isHot = true;

        return res;
}

static bool textInputLogic(unsigned int id, bool over){
    //same as button logic without the react on left up
    bool res = false;
    // process down
    if (!anyActive())
    {
        if (over)
            setHot(id);
        if (isHot(id) && g_state.leftPressed)
            setInputable(id);
    }

    if (isHot(id))
        g_state.isHot = true;

    return res;
}

static void updateInput(int mx, int my, unsigned char mbut, int scroll, char asciiCode)
{
        bool left = (mbut & IMGUI_MBUT_LEFT) != 0;

        g_state.mx = mx;
        g_state.my = my;
        g_state.leftPressed = !g_state.left && left;
        g_state.leftReleased = g_state.left && !left;
        g_state.left = left;

        g_state.scroll = scroll;

        if(asciiCode > 0x80) //only ascii code handled
            asciiCode = 0;
        g_state.lastAscii = g_state.ascii;
        g_state.ascii = asciiCode;
}

void imguiBeginFrame(int mx, int my, unsigned char mbut, int scroll, char asciiCode/*=0*/)
{
        imguiResetCaret();
        imguiResetColors();
        imguiResetEnables();

        updateInput(mx,my,mbut,scroll,asciiCode);

        g_state.hot = g_state.hotToBe;
        g_state.hotToBe = 0;

        g_state.wentActive = false;
        g_state.isActive = false;
        g_state.isHot = false;

        g_state.widgetX = 0;
        g_state.widgetY = 0;
        g_state.widgetW = 0;
        g_state.clear();
        g_state.push();

        g_state.areaId = 1;
        g_state.widgetId = 1;

        resetGfxCmdQueue();
}

void imguiEndFrame()
{
        clearInput();
}

const imguiGfxCmd* imguiGetRenderQueue()
{
        return g_gfxCmdQueue;
}

int imguiGetRenderQueueSize()
{
        return g_gfxCmdQueueSize;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static const int BUTTON_HEIGHT = 20;
static const int SLIDER_HEIGHT = 20;
static const int SLIDER_MARKER_WIDTH = 10;
static const int CHECK_SIZE = 8;
static const int DEFAULT_SPACING = 4;
static const int TEXT_HEIGHT = 8;
static const int SCROLL_AREA_PADDING = 6;
static const int INDENT_SIZE = 16;
static const int AREA_HEADER = 28;

static int g_scrollTop = 0;
static int g_scrollBottom = 0;
static int g_scrollRight = 0;
static int g_scrollAreaTop = 0;
static int* g_scrollVal = 0;
static int g_focusTop = 0;
static int g_focusBottom = 0;
static unsigned int g_scrollId = 0;
static bool g_insideScrollArea = false;

bool imguiBeginScrollArea(const char* name, int x, int y, int w, int h, int* scroll)
{
        g_state.areaId++;
        g_state.widgetId = 0;
        g_scrollId = (g_state.areaId<<16) | g_state.widgetId;

        g_state.widgetX = x + SCROLL_AREA_PADDING;
        g_state.widgetY = y+h-AREA_HEADER + (scroll ? *scroll : 0); // @rlyeh: support for fixed areas
        g_state.widgetW = w - SCROLL_AREA_PADDING*4;
        g_state.push();
        g_scrollTop = y-AREA_HEADER+h;
        g_scrollBottom = y+SCROLL_AREA_PADDING;
        g_scrollRight = x+w - SCROLL_AREA_PADDING*3;
        g_scrollVal = scroll;

        g_scrollAreaTop = g_state.widgetY;

        g_focusTop = y-AREA_HEADER;
        g_focusBottom = y-AREA_HEADER+h;

        g_insideScrollArea = inRect(x, y, w, h, false);
        g_state.insideCurrentScroll = g_insideScrollArea;

        addGfxCmdRoundedRect((float)x, (float)y, (float)w, (float)h, 6, black_alpha(192) );

        addGfxCmdText(x+AREA_HEADER/2, y+h-AREA_HEADER/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, name, white_alpha(128) );

        if( g_scrollVal ) { // @rlyeh: support for fixed areas
              addGfxCmdScissor(
              x < 0 ? 0 : x+SCROLL_AREA_PADDING, //@rlyeh: fix scissor clipping problem when scroll area is on left rect client
              y+SCROLL_AREA_PADDING,
              w-SCROLL_AREA_PADDING*4 + ( x < 0 ? x : 0 ),   // @rlyeh: small optimization; @todo: on the right as well
              h > (AREA_HEADER + SCROLL_AREA_PADDING ) ? h - (AREA_HEADER + SCROLL_AREA_PADDING) : h ); // @rlyeh: fix when collapsing areas ( h <= AREA_HEADER )
        }

        return g_insideScrollArea;
}

void imguiEndScrollArea()
{
        if( g_scrollVal ) { // @rlyeh: support for fixed areas
            // Disable scissoring.
            addGfxCmdScissor(-1,-1,-1,-1);
        }

        // Draw scroll bar
        int x = g_scrollRight+SCROLL_AREA_PADDING/2;
        int y = g_scrollBottom;
        int w = SCROLL_AREA_PADDING*2;
        int h = g_scrollTop - g_scrollBottom;

        int stop = g_scrollAreaTop;
        int sbot = g_state.widgetY;
        int sh = stop - sbot; // The scrollable area height.

        float barHeight = (float)h/(float)sh;

        if (h > AREA_HEADER && barHeight < 1) // @rlyeh: fix when area size is too small
        {
                float barY = (float)(y - sbot)/(float)sh;
                if (barY < 0) barY = 0;
                if (barY > 1) barY = 1;

                // Handle scroll bar logic.
                unsigned int hid = g_scrollId;
                int hx = x;
                int hy = y + (int)(barY*h);
                int hw = w;
                int hh = (int)(barHeight*h);

                const int range = h - (hh-1);
                bool over = inRect(hx, hy, hw, hh, true);
                buttonLogic(hid, over);
                float u = (float)(hy-y) / (float)range;
                if (isActive(hid))
                {
                        if (g_state.wentActive)
                        {
                                g_state.dragY = g_state.my;
                                g_state.dragOrig = u;
                        }
                        if (g_state.dragY != g_state.my)
                        {
                                u = g_state.dragOrig + (g_state.my - g_state.dragY) / (float)range;
                                if (u < 0) u = 0;
                                if (u > 1) u = 1;
                                *g_scrollVal = (int)((1-u) * (sh - h));
                        }
                }
                else if(u<=0||u>1) *g_scrollVal = (int)(sh-h);  //rlyeh: @fix when resizing windows

                // BG
                addGfxCmdRoundedRect((float)x, (float)y, (float)w, (float)h, (float)w/2-1, black_alpha(196) );
                // Bar
                if (isActive(hid))
                        addGfxCmdRoundedRect((float)hx, (float)hy, (float)hw, (float)hh, (float)w/2-1, theme_alpha(196) );
                else
                        addGfxCmdRoundedRect((float)hx, (float)hy, (float)hw, (float)hh, (float)w/2-1, isHot(hid) ? theme_alpha(96) : theme_alpha(64) );

                // Handle mouse scrolling.
                if (g_insideScrollArea) // && !anyActive())
                {
                        if (g_state.scroll)
                        {
                                *g_scrollVal += 20*g_state.scroll;
                                if (*g_scrollVal < 0) *g_scrollVal = 0;
                                if (*g_scrollVal > (sh - h)) *g_scrollVal = (sh - h);
                        }
                }
        }
        else *g_scrollVal = 0; // @rlyeh: fix for mismatching scroll when collapsing/uncollapsing content larger than container height
        g_state.insideCurrentScroll = false;
}

bool imguiButton(const char* text)
{
        g_state.widgetId++;
        unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

        int x = g_state.widgetX;
        int y = g_state.widgetY - BUTTON_HEIGHT;
        int w = g_state.widgetW;
        int h = BUTTON_HEIGHT;

        int offset = w - DEFAULT_SPACING;
        g_state.widgetX += offset;
        g_state.widgetW -= offset;
        g_state.push();
        g_state.widgetX -= offset;
        g_state.widgetW += offset;
        g_state.push();
        g_state.widgetY -= BUTTON_HEIGHT + DEFAULT_SPACING;
        g_state.push();

        bool over = enabled && inRect(x, y, w, h, true);
        bool res = buttonLogic(id, over);

        addGfxCmdRoundedRect((float)x, (float)y, (float)w, (float)h, (float)BUTTON_HEIGHT/2-1, isActive(id) ? gray_alpha(196) : gray_alpha(96) );
        if (enabled)
                addGfxCmdText(x+w/2, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_CENTER|IMGUI_ALIGN_BASELINE, text, isHot(id) ? theme_alpha(256) : theme_alpha(192) );
        else
                addGfxCmdText(x+w/2, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_CENTER|IMGUI_ALIGN_BASELINE, text, gray_alpha(200) );

        return res;
}

bool imguiItem(const char* text)
{
        g_state.widgetId++;
        unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

        int x = g_state.widgetX;
        int y = g_state.widgetY - BUTTON_HEIGHT;
        int w = g_state.widgetW;
        int h = BUTTON_HEIGHT;
        g_state.widgetY -= BUTTON_HEIGHT + DEFAULT_SPACING;
        g_state.push();

        bool over = enabled && inRect(x, y, w, h, true);
        bool res = buttonLogic(id, over);

        if (isHot(id))
                addGfxCmdRoundedRect((float)x, (float)y, (float)w, (float)h, 2.0f, isActive(id) ? theme_alpha(192) : theme_alpha(96) );

        if (enabled)
                addGfxCmdText(x+BUTTON_HEIGHT/2, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, theme_alpha(200) );
        else
                addGfxCmdText(x+BUTTON_HEIGHT/2, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, gray_alpha(200) );

        return res;
}

bool imguiText(const char* text)
{
        g_state.widgetId++;
        unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

        int x = g_state.widgetX;
        int y = g_state.widgetY - BUTTON_HEIGHT;
        int w = g_state.widgetW;
        int h = BUTTON_HEIGHT;
        g_state.widgetY -= BUTTON_HEIGHT + DEFAULT_SPACING;
        g_state.push();

        bool over = enabled && inRect(x, y, w, h, true);
        bool res = buttonLogic(id, over);

        if (enabled)
                addGfxCmdText(x+BUTTON_HEIGHT/2, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, isHot(id) ? theme_alpha(200) : theme_alpha(192) );
        else
                addGfxCmdText(x+BUTTON_HEIGHT/2, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, gray_alpha(200) );

        return res;
}

bool imguiCheck(const char* text, bool checked)
{
        g_state.widgetId++;
        unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

        int x = g_state.widgetX;
        int y = g_state.widgetY - BUTTON_HEIGHT;
        int w = g_state.widgetW;
        int h = BUTTON_HEIGHT;
        g_state.widgetY -= BUTTON_HEIGHT + DEFAULT_SPACING;
        g_state.push();

        bool over = enabled && inRect(x, y, w, h, true);
        bool res = buttonLogic(id, over);

        const int cx = x+BUTTON_HEIGHT/2-CHECK_SIZE/2;
        const int cy = y+BUTTON_HEIGHT/2-CHECK_SIZE/2;
        addGfxCmdRoundedRect((float)cx-3, (float)cy-3, (float)CHECK_SIZE+6, (float)CHECK_SIZE+6, 4,  isActive(id) ? gray_alpha(196) : gray_alpha(96) );
        if (checked)
        {
                if (enabled)
                        addGfxCmdRoundedRect((float)cx, (float)cy, (float)CHECK_SIZE, (float)CHECK_SIZE, (float)CHECK_SIZE/2-1, isActive(id) ? theme_alpha(256) : theme_alpha(200));
                else
                        addGfxCmdRoundedRect((float)cx, (float)cy, (float)CHECK_SIZE, (float)CHECK_SIZE, (float)CHECK_SIZE/2-1, gray_alpha(200) );
        }

        if (enabled)
                addGfxCmdText(x+BUTTON_HEIGHT, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, isHot(id) ? theme_alpha(256) : theme_alpha(200) );
        else
                addGfxCmdText(x+BUTTON_HEIGHT, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, gray_alpha(200) );

        return res;
}

bool imguiCollapse(const char* text, const char* subtext, bool checked)
{
        g_state.widgetId++;
        unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

        int x = g_state.widgetX;
        int y = g_state.widgetY - BUTTON_HEIGHT;
        int w = g_state.widgetW;
        int h = BUTTON_HEIGHT;
        g_state.widgetY -= BUTTON_HEIGHT; // + DEFAULT_SPACING;
        g_state.push();

        const int cx = x+BUTTON_HEIGHT/2-CHECK_SIZE/2;
        const int cy = y+BUTTON_HEIGHT/2-CHECK_SIZE/2;

        bool over = enabled && inRect(x, y, w, h, true);
        bool res = buttonLogic(id, over);

        if (checked)
                addGfxCmdTriangle(cx, cy, CHECK_SIZE, CHECK_SIZE, 2, isActive(id) ? theme_alpha(256) : theme_alpha(192) );
        else
                addGfxCmdTriangle(cx, cy, CHECK_SIZE, CHECK_SIZE, 1, isActive(id) ? theme_alpha(256) : theme_alpha(192) );

        if (enabled)
                addGfxCmdText(x+BUTTON_HEIGHT, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, isHot(id) ? theme_alpha(256) : theme_alpha(192) );
        else
                addGfxCmdText(x+BUTTON_HEIGHT, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, gray_alpha(192) );

        if (subtext)
                addGfxCmdText(x+w-BUTTON_HEIGHT/2, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_RIGHT|IMGUI_ALIGN_BASELINE, subtext, theme_alpha(128));

        return res;
}

void imguiLabel(const char* text)
{
        int x = g_state.widgetX;
        int y = g_state.widgetY - BUTTON_HEIGHT;
        g_state.widgetY -= BUTTON_HEIGHT;
        g_state.push();
        addGfxCmdText(x, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, theme_alpha(256) );
}

void imguiValue(const char* text)
{
        const int x = g_state.widgetX;
        const int y = g_state.widgetY - BUTTON_HEIGHT;
        const int w = g_state.widgetW;
        g_state.widgetY -= BUTTON_HEIGHT;
        g_state.push();
        addGfxCmdText(x+w-BUTTON_HEIGHT/2, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_RIGHT|IMGUI_ALIGN_BASELINE, text, theme_alpha(192) );
}

bool imguiSlider(const char* text, float* val, float vmin, float vmax, float vinc, const char *format)
{
        g_state.widgetId++;
        unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

        int x = g_state.widgetX;
        int y = g_state.widgetY - BUTTON_HEIGHT;
        int w = g_state.widgetW;
        int h = SLIDER_HEIGHT;
        g_state.widgetY -= SLIDER_HEIGHT + DEFAULT_SPACING;
        g_state.push();

        addGfxCmdRoundedRect((float)x, (float)y, (float)w, (float)h, 4.0f, black_alpha(96) );

        const int range = w - SLIDER_MARKER_WIDTH;

        float u = (*val - vmin) / (vmax-vmin);
        if (u < 0) u = 0;
        if (u > 1) u = 1;
        int m = (int)(u * range);

        bool over = enabled && inRect(x+m, y, SLIDER_MARKER_WIDTH, SLIDER_HEIGHT, true);
        bool res = buttonLogic(id, over);
        bool valChanged = false;

        if (isActive(id))
        {
                if (g_state.wentActive)
                {
                        g_state.dragX = g_state.mx;
                        g_state.dragOrig = u;
                }
                if (g_state.dragX != g_state.mx)
                {
                        u = g_state.dragOrig + (float)(g_state.mx - g_state.dragX) / (float)range;
                        if (u < 0) u = 0;
                        if (u > 1) u = 1;
                        *val = vmin + u*(vmax-vmin);
                        *val = floorf(*val/vinc+0.5f)*vinc; // Snap to vinc
                        m = (int)(u * range);
                        valChanged = true;
                }
        }

        unsigned int col = gray_alpha(64);
        if( enabled ) {
            if (isActive(id)) col = theme_alpha(256);
            else col = isHot(id) ? theme_alpha(128) : theme_alpha(64);
        }
        addGfxCmdRoundedRect((float)(x+m), (float)y, (float)SLIDER_MARKER_WIDTH, (float)SLIDER_HEIGHT, 4.0f, col );

        // TODO: fix this, take a look at 'nicenum'.
        int digits = (int)(std::ceilf(std::log10f(vinc)));
        char msg[128];
        const char *replaced = replace_str( format, "%d", "%.*f" );
        sprintf(msg, replaced ? replaced : "%.*f", digits >= 0 ? 0 : -digits, *val);
        if( replaced ) {
            free( (void *)replaced );
        }

        if (enabled)
        {
                addGfxCmdText(x+SLIDER_HEIGHT/2, y+SLIDER_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, isHot(id) | isActive(id) ? theme_alpha(256) : theme_alpha(192) ); // @rlyeh: fix blinking colours
                addGfxCmdText(x+w-SLIDER_HEIGHT/2, y+SLIDER_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_RIGHT|IMGUI_ALIGN_BASELINE, msg, isHot(id) | isActive(id) ? theme_alpha(256) : theme_alpha(192) ); // @rlyeh: fix blinking colours
        }
        else
        {
                addGfxCmdText(x+SLIDER_HEIGHT/2, y+SLIDER_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, gray_alpha(192) );
                addGfxCmdText(x+w-SLIDER_HEIGHT/2, y+SLIDER_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_RIGHT|IMGUI_ALIGN_BASELINE, msg, gray_alpha(192) );
        }

        return res || valChanged;
}

bool imguiRange(const char* text, float* val0, float *val1, float vmin, float vmax, float vinc, const char *format)
{
        int x = g_state.widgetX;
        int y = g_state.widgetY - BUTTON_HEIGHT;
        int w = g_state.widgetW;
        int h = SLIDER_HEIGHT;
        g_state.widgetY -= SLIDER_HEIGHT + DEFAULT_SPACING;
        g_state.push();

// dims

        if(  vmin >  vmax ) { float swap = vmin;   vmin = vmax;   vmax = swap; }
        if( *val0 > *val1 ) { float swap = *val0; *val0 = *val1; *val1 = swap; }
        if( *val0 <  vmin ) { *val0 = vmin; }
        if( *val1 >  vmax ) { *val1 = vmax; }

        const int range = w - SLIDER_MARKER_WIDTH;

        float u0 = (*val0 - vmin) / (vmax-vmin);
        if (u0 < 0) u0 = 0;
        if (u0 > 1) u0 = 1;
        int m0 = (int)(u0 * range);

        float u1 = (*val1 - vmin) / (vmax-vmin);
        if (u1 < 0) u1 = 0;
        if (u1 > 1) u1 = 1;
        int m1 = (int)(u1 * range);

// button

        addGfxCmdRoundedRect((float)x + m0, (float)y, (float)m1 - m0 + SLIDER_MARKER_WIDTH, (float)h, 4.0f, enabled ? theme_alpha(64) : gray_alpha(64) );
        addGfxCmdRoundedRect((float)x, (float)y, (float)w, (float)h, 4.0f, black_alpha(96) );

        bool is_highlighted = false;
        bool is_res = false;
        bool has_changed = false;

// slide #0
{
        float *val = val0;
        float u = u0, m = m0;

        g_state.widgetId++;
        unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

        bool over = enabled && inRect(x+m, y, SLIDER_MARKER_WIDTH, SLIDER_HEIGHT, true);
        bool res = buttonLogic(id, over);
        bool valChanged = false;

        if (isActive(id))
        {
                if (g_state.wentActive)
                {
                        g_state.dragX = g_state.mx;
                        g_state.dragOrig = u;
                }
                if (g_state.dragX != g_state.mx)
                {
                        u = g_state.dragOrig + (float)(g_state.mx - g_state.dragX) / (float)range;
                        if (u < 0) u = 0;
                        if (u > 1) u = 1;
                        *val = vmin + u*(vmax-vmin);
                        *val = floorf(*val/vinc+0.5f)*vinc; // Snap to vinc
                        m = (int)(u * range);
                        valChanged = true;
                }
        }

        unsigned int col = gray_alpha(64);
        if( enabled ) {
            if (isActive(id)) col = theme_alpha(256);
            else col = isHot(id) ? theme_alpha(128) : theme_alpha(64);
        }
        addGfxCmdRoundedRect((float)(x+m), (float)y, (float)SLIDER_MARKER_WIDTH, (float)SLIDER_HEIGHT, 4.0f, col );

        is_highlighted |= ( isHot(id) | isActive(id) );
        is_res |= res;
        has_changed |= valChanged;
}

// slide #1
{
        float *val = val1;
        float u = u1, m = m1;

        g_state.widgetId++;
        unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

        bool over = enabled && inRect(x+m, y, SLIDER_MARKER_WIDTH, SLIDER_HEIGHT, true);
        bool res = buttonLogic(id, over);
        bool valChanged = false;

        if (isActive(id))
        {
                if (g_state.wentActive)
                {
                        g_state.dragX = g_state.mx;
                        g_state.dragOrig = u;
                }
                if (g_state.dragX != g_state.mx)
                {
                        u = g_state.dragOrig + (float)(g_state.mx - g_state.dragX) / (float)range;
                        if (u < 0) u = 0;
                        if (u > 1) u = 1;
                        *val = vmin + u*(vmax-vmin);
                        *val = floorf(*val/vinc+0.5f)*vinc; // Snap to vinc
                        m = (int)(u * range);
                        valChanged = true;
                }
        }

        unsigned int col = gray_alpha(64);
        if( enabled ) {
            if (isActive(id)) col = theme_alpha(256);
            else col = isHot(id) ? theme_alpha(128) : theme_alpha(64);
        }
        addGfxCmdRoundedRect((float)(x+m), (float)y, (float)SLIDER_MARKER_WIDTH, (float)SLIDER_HEIGHT, 4.0f, col );

        is_highlighted |= ( isHot(id) | isActive(id) );
        is_res |= res;
        has_changed |= valChanged;
}

// text

        // TODO: fix this, take a look at 'nicenum'.
        int digits = (int)(std::ceilf(std::log10f(vinc)));
        char msg[128];
        const char *replaced = replace_str( format, "%d", "%.*f" );
        sprintf(msg, replaced ? replaced : "%.*f - %.*f", digits >= 0 ? 0 : -digits, *val0, digits >= 0 ? 0 : -digits, *val1);
        if( replaced ) {
            free( (void *)replaced );
        }

        if (enabled)
        {
                addGfxCmdText(x+SLIDER_HEIGHT/2, y+SLIDER_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, is_highlighted ? theme_alpha(256) : theme_alpha(192) ); // @rlyeh: fix blinking colours
                addGfxCmdText(x+w-SLIDER_HEIGHT/2, y+SLIDER_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_RIGHT|IMGUI_ALIGN_BASELINE, msg, is_highlighted ? theme_alpha(256) : theme_alpha(192) ); // @rlyeh: fix blinking colours
        }
        else
        {
                addGfxCmdText(x+SLIDER_HEIGHT/2, y+SLIDER_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, gray_alpha(192) );
                addGfxCmdText(x+w-SLIDER_HEIGHT/2, y+SLIDER_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_RIGHT|IMGUI_ALIGN_BASELINE, msg, gray_alpha(192) );
        }

        return is_res || has_changed;
}

bool imguiTextInput(const char* text, char* buffer, unsigned int bufferLength)
{
    bool res = true;
    //--
    //Handle label
    g_state.widgetId++;
    unsigned int id = (g_state.areaId<<16) | g_state.widgetId;
    int x = g_state.widgetX;
    int y = g_state.widgetY - BUTTON_HEIGHT;
    addGfxCmdText(x, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, enabled ? white_alpha(255) : gray_alpha(128));
    unsigned int textLen = (unsigned int)( imguiCalcText( text ) );
    //--
    //Handle keyboard input if any
    unsigned int L = strlen(buffer);
    if( enabled ) {
        if(isInputable(id) && g_state.ascii == 0x08 && g_state.ascii!=g_state.lastAscii)//backspace
        {    if(L>0 && buffer[L-1]>8) buffer[L-1]=0; }
        else if(isInputable(id) && g_state.ascii == 0x0D && g_state.ascii!=g_state.lastAscii)//enter
            g_state.inputable = 0;
        else if(isInputable(id) && L < bufferLength-1 && g_state.ascii!=0 && g_state.ascii!=g_state.lastAscii){
            ++L;
            buffer[L-1] = g_state.ascii;
            buffer[L] = 0;
        }
    }
    //--
    //Handle buffer data
    x+=textLen;
    int w = g_state.widgetW-textLen;
    int h = BUTTON_HEIGHT;
    bool over = inRect(x, y, w, h);
    res = textInputLogic(id, over);
    if( enabled ) {
        char _buffer[32];
        strcpy( _buffer, buffer );
        if( isInputable(id) && caret ) { _buffer[L] = '|'; _buffer[L+1] = 0; }
        addGfxCmdRoundedRect((float)x, (float)y, (float)w, (float)h, (float)BUTTON_HEIGHT/2-1, isInputable(id) ? theme_alpha(256):gray_alpha(96));
        addGfxCmdText(x+7, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, _buffer, isInputable(id) ? black_alpha(256): white_alpha(256));
    } else {
        addGfxCmdRoundedRect((float)x, (float)y, (float)w, (float)h, (float)BUTTON_HEIGHT/2-1, gray_alpha(64));
        addGfxCmdText(x+7, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, buffer, white_alpha(128));
    }
    //--
    g_state.widgetY -= BUTTON_HEIGHT + DEFAULT_SPACING;
    g_state.push();
    return res;
}

void imguiPair(const char* text, const char *value)  // @rlyeh: new widget
{
    imguiLabel(text);
    g_state.widgetY += BUTTON_HEIGHT;
    g_state.push();
    imguiValue(value);
}

bool imguiList(const char* text, size_t n_options, const char** options, int &choosing, int &clicked) // @rlyeh: new widget
{
    g_state.widgetId++;
    unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

    int x = g_state.widgetX;
    int y = g_state.widgetY - BUTTON_HEIGHT;
    int w = g_state.widgetW;
    int h = BUTTON_HEIGHT;
    g_state.widgetY -= BUTTON_HEIGHT; // + DEFAULT_SPACING;
    g_state.push();

    const int cx = x+BUTTON_HEIGHT/2-CHECK_SIZE/2;
    const int cy = y+BUTTON_HEIGHT/2-CHECK_SIZE/2;

    bool over = enabled && inRect(x, y, w, h, true);
    bool res = buttonLogic(id, over);

    if (enabled)
    {
        addGfxCmdRoundedRect((float)x, (float)y, (float)w, (float)h, 2.0f, !isHot(id) ? gray_alpha(64) : (isActive(id) ? theme_alpha(192) : theme_alpha(choosing ? 64 : 96)) );

        addGfxCmdTriangle((float)cx, (float)cy, CHECK_SIZE, CHECK_SIZE, choosing ? 2 : 1, isActive(id) ? theme_alpha(256) : theme_alpha(192) );

        addGfxCmdText(x+BUTTON_HEIGHT, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, clicked < 0 ? text : options[clicked], theme_alpha(192) );

        //addGfxCmdRoundedRect(x+w-BUTTON_HEIGHT/2, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, (float)CHECK_SIZE, (float)CHECK_SIZE, (float)CHECK_SIZE/2-1, isActive(id) ? theme_alpha(256) : theme_alpha(192));

        if( res )
            choosing ^= 1;
    }
    else
    {
        addGfxCmdRoundedRect((float)x, (float)y, (float)w, (float)h, 2.0f, gray_alpha(64) );

        addGfxCmdTriangle((float)cx, (float)cy, CHECK_SIZE, CHECK_SIZE, choosing ? 2 : 1, isActive(id) ? theme_alpha(256) : theme_alpha(192) );

        addGfxCmdText(x+BUTTON_HEIGHT, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, clicked < 0 ? text : options[clicked], gray_alpha(192) );

        //addGfxCmdRoundedRect(x+w-BUTTON_HEIGHT/2, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, (float)CHECK_SIZE, (float)CHECK_SIZE, (float)CHECK_SIZE/2-1, isActive(id) ? theme_alpha(256) : theme_alpha(192));
    }

    bool result = false;

    if( choosing )
    {
        // choice selector
        imguiIndent();
            // hotness = are we on focus?
            bool hotness = isHot(id) | isActive(id);
            // choice selector list
            for( size_t n = 0; !result && n < n_options; ++n )
            {
                // got answer?
                if( imguiItem( options[n] ) )
                    clicked = n, choosing = 0, result = true;

                unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

                // ensure that widget is still on focus while choosing
                hotness |= isHot(id) | isActive(id);
            }
            // close on blur
            if( !hotness && anyHot() )
            {}//    choosing = 0;
        imguiUnindent();
    }

    return result;
}

bool imguiRadio(const char* text, size_t n_options, const char** options, int &clicked) // @rlyeh: new widget
{
    g_state.widgetId++;
    unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

    int x = g_state.widgetX;
    int y = g_state.widgetY - BUTTON_HEIGHT;
    int w = g_state.widgetW;
    int h = BUTTON_HEIGHT;
    g_state.widgetY -= BUTTON_HEIGHT; // + DEFAULT_SPACING;
    g_state.push();

    const int cx = x+BUTTON_HEIGHT/2-CHECK_SIZE/2;
    const int cy = y+BUTTON_HEIGHT/2-CHECK_SIZE/2;

    bool over = enabled && inRect(x, y, w, h, true);
    bool res = buttonLogic(id, over);

    if (enabled)
            addGfxCmdText(cx, cy, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, isHot(id) ? theme_alpha(256) : theme_alpha(192) );
    else
            addGfxCmdText(cx, cy, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, gray_alpha(192) );

    bool result = false;

    imguiIndent();
        for( size_t i = 0; i < n_options; ++i )
        {
            bool cl = ( clicked == i );
            if( imguiCheck( options[i], cl ) )
                clicked = i, result = true;
        }
    imguiUnindent();

    return result;
}

void imguiProgressBar(const char* text, float val, bool show_decimals)
{
    const float vmin = 0.00f, vmax = 100.00f;

    if( val < 0.f ) val = 0.f; else if( val > 100.f ) val = 100.f;

    g_state.widgetId++;
    unsigned int id = (g_state.areaId<<16) | g_state.widgetId;

    int x = g_state.widgetX;
    int y = g_state.widgetY - BUTTON_HEIGHT;
    int w = g_state.widgetW;
    int h = SLIDER_HEIGHT;
    g_state.widgetY -= SLIDER_HEIGHT + DEFAULT_SPACING;
    g_state.push();

    addGfxCmdRoundedRect((float)x, (float)y, (float)w, (float)h, 4.0f, black_alpha(96) );

    const int range = w - SLIDER_MARKER_WIDTH;

    float u = (val - vmin) / (vmax-vmin);
    if (u < 0) u = 0;
    if (u > 1) u = 1;
    int m = (int)(u * range);

    addGfxCmdRoundedRect((float)(x+0), (float)y, (float)(SLIDER_MARKER_WIDTH+m), (float)SLIDER_HEIGHT, 4.0f, theme_alpha(64) );

    // TODO: fix this, take a look at 'nicenum'.
    int digits = (int)(std::ceilf(std::log10f(0.01f)));
    char msg[128];
    if( show_decimals )
    sprintf(msg, "%.*f%%", digits >= 0 ? 0 : -digits, val);
    else
    sprintf(msg, "%d%%", int(val) );

    addGfxCmdText(x+SLIDER_HEIGHT/2, y+SLIDER_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, theme_alpha(192) );
    addGfxCmdText(x+w-SLIDER_HEIGHT/2, y+SLIDER_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_RIGHT|IMGUI_ALIGN_BASELINE, msg, theme_alpha(192) );
}

bool imguiBitmask(const char* text, unsigned *mask, int bits)
{
    int x = g_state.widgetX;
    int y = g_state.widgetY - BUTTON_HEIGHT;
    int w = g_state.widgetW;
    int h = BUTTON_HEIGHT;

    //--
    //Handle label
    g_state.widgetId++;
    unsigned int id = (g_state.areaId<<16) | g_state.widgetId;
//    addGfxCmdText(x, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, white_alpha(255));
    if (enabled)
        addGfxCmdText(x+DEFAULT_SPACING, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, isHot(id) ? theme_alpha(256) : theme_alpha(200) );
    else
        addGfxCmdText(x+DEFAULT_SPACING, y+BUTTON_HEIGHT/2-TEXT_HEIGHT/2, IMGUI_ALIGN_LEFT|IMGUI_ALIGN_BASELINE, text, gray_alpha(200) );
    unsigned int textLen = (unsigned int)( imguiCalcText(text) );
    //--

        bool ress = false;
        unsigned before = *mask;

        const int cxx = x + textLen + ( textLen > 0 ? 1 : 0 ) * DEFAULT_SPACING + CHECK_SIZE/2;
        const int cy = y+BUTTON_HEIGHT/2-CHECK_SIZE/2;

int offset = (cxx - x) + bits * (CHECK_SIZE+6);
g_state.widgetX += offset;
g_state.widgetW -= offset;
g_state.push();
g_state.widgetX -= offset;
g_state.widgetW += offset;
g_state.push();
g_state.widgetY -= BUTTON_HEIGHT + DEFAULT_SPACING;
g_state.push();

        for( int i = 0; i < bits; ++i ) { //bits; i-- > 0; ) {

            int cx = cxx + (bits-1-i) * (CHECK_SIZE+6);
            bool checked = ((*mask) & (1<<i)) == (1<<i);

            g_state.widgetId++;
            unsigned int id = (g_state.areaId<<16) | g_state.widgetId;
            bool over = enabled && inRect((float)cx-3, (float)cy-3, (float)CHECK_SIZE+6, (float)CHECK_SIZE+6, true);
            bool res = buttonLogic(id, over);

            if( res ) {
                ress |= res;
                (*mask) ^= (1<<i);
            }

            //addGfxCmdRoundedRect((float)cx-3, (float)cy-3, (float)CHECK_SIZE+6, (float)CHECK_SIZE+6, 4,  isActive(id) ? gray_alpha(196) : gray_alpha(96) );
            if (checked)
            {
                    if (enabled)
                            addGfxCmdRoundedRect((float)cx, (float)cy, (float)CHECK_SIZE, (float)CHECK_SIZE, (float)CHECK_SIZE/2-1, isActive(id) ? theme_alpha(256) : theme_alpha(200));
                    else
                            addGfxCmdRoundedRect((float)cx, (float)cy, (float)CHECK_SIZE, (float)CHECK_SIZE, (float)CHECK_SIZE/2-1, gray_alpha(200) );
            } else {
                    if (enabled)
                            addGfxCmdRoundedRect((float)cx+CHECK_SIZE/4, (float)cy+CHECK_SIZE/4, (float)CHECK_SIZE/2, (float)CHECK_SIZE/2, (float)CHECK_SIZE/8-1, isActive(id) ? theme_alpha(256) : theme_alpha(200));
                    else
                            addGfxCmdRoundedRect((float)cx+CHECK_SIZE/4, (float)cy+CHECK_SIZE/4, (float)CHECK_SIZE/2, (float)CHECK_SIZE/2, (float)CHECK_SIZE/8-1, gray_alpha(200) );
            }
        }

        return ress | (before != *mask);
}

void imguiIndent()
{
        g_state.widgetX += INDENT_SIZE;
        g_state.widgetW -= INDENT_SIZE;
        g_state.push();
}

void imguiUnindent()
{
        g_state.widgetX -= INDENT_SIZE;
        g_state.widgetW += INDENT_SIZE;
        g_state.push();
}

void imguiSeparator()
{
        g_state.widgetY -= DEFAULT_SPACING*3;
        g_state.push();
}

void imguiSeparatorLine()
{
        int x = g_state.widgetX;
        int y = g_state.widgetY - DEFAULT_SPACING*2;
        int w = g_state.widgetW;
        int h = 1;
        g_state.widgetY -= DEFAULT_SPACING*4;
        g_state.push();

        addGfxCmdRect((float)x, (float)y, (float)w, (float)h, theme_alpha(32) );
}

// @todo: fixme, buggy
void imguiTabulator()
{
    const int BUTTON_WIDTH = g_state.widgetW > g_state.widgetX ? g_state.widgetW - g_state.widgetX : g_state.widgetX - g_state.widgetW;

    g_state.widgetX += BUTTON_WIDTH;
    g_state.widgetW += BUTTON_WIDTH;

    // should in fact get retrieved from last widget queued size
    g_state.widgetY += BUTTON_HEIGHT;
    g_state.push();
}

// @todo: fixme, buggy
void imguiTabulatorLine()
{
    int x = g_state.widgetX + DEFAULT_SPACING*2;
    int y = g_state.widgetY;
    int w = 1;
    int h = 100; //g_state.widgetH;

    g_state.widgetX += DEFAULT_SPACING*4;
    g_state.widgetW += DEFAULT_SPACING*4;
    g_state.push();

    addGfxCmdRect((float)x, (float)y, (float)w, (float)h, theme_alpha(32) );
}

void imguiDrawText(int x, int y, imguiTextAlign align, const char* text, unsigned int color)
{
        addGfxCmdText(x, y, align, text, color);
}

void imguiDrawLine(float x0, float y0, float x1, float y1, float r, unsigned int color)
{
        addGfxCmdLine(x0, y0, x1, y1, r, color);
}

void imguiDrawRect(float x, float y, float w, float h, unsigned int color)
{
        addGfxCmdRect(x, y, w, h, color);
}

void imguiDrawRoundedRect(float x, float y, float w, float h, float r, unsigned int color)
{
        addGfxCmdRoundedRect(x, y, w, h, r, color);
}

int imguiCalcText( const char *text )
{
    return (*imguiRenderCalcText)(text);
}

static float imgui__modf(float a, float b) { return fmodf(a, b); }
static float imgui__clampf(float a, float mn, float mx) { return a < mn ? mn : (a > mx ? mx : a); }
static float imgui__hue(float h, float m1, float m2) {
    if (h < 0) h += 1;
    if (h > 1) h -= 1;
    if (h < 1.0f/6.0f)
        return m1 + (m2 - m1) * h * 6.0f;
    else if (h < 3.0f/6.0f)
        return m2;
    else if (h < 4.0f/6.0f)
        return m1 + (m2 - m1) * (2.0f/3.0f - h) * 6.0f;
    return m1;
}

unsigned int imguiHSLA(float h, float s, float l, unsigned char a)
{
    float m1, m2;
    unsigned char r,g,b;
    h = imgui__modf(h, 1.0f);
    if (h < 0.0f) h += 1.0f;
    s = imgui__clampf(s, 0.0f, 1.0f);
    l = imgui__clampf(l, 0.0f, 1.0f);
    m2 = l <= 0.5f ? (l * (1 + s)) : (l + s - l * s);
    m1 = 2 * l - m2;
    r = (unsigned char)imgui__clampf(imgui__hue(h + 1.0f/3.0f, m1, m2) * 255.0f, 0, 255);
    g = (unsigned char)imgui__clampf(imgui__hue(h, m1, m2) * 255.0f, 0, 255);
    b = (unsigned char)imgui__clampf(imgui__hue(h - 1.0f/3.0f, m1, m2) * 255.0f, 0, 255);
    return imguiRGBA(r,g,b,a);
}

unsigned int imguiRGBA(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
    return (r) | (g << 8) | (b << 16) | (a << 24);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int imguiShowDialog( const char *text, int *answer ) {
    bool clicked = false;

    imguiLabel( text );
    imguiSpaceDiv();
        if( imguiButton("yes") ) {
            clicked = true;
            *answer = true;
        }
        int pos = imguiStackSet(-1);
            imguiSpaceShift();
                if( imguiButton("no") ) {
                    clicked = true;
                    *answer = false;
                }
            imguiSpaceUnshift();
        imguiStackSet(pos);
    imguiSpaceMul();

    return clicked;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
#   pragma warning(pop) // C4996
#endif
