import { definePatch } from "../modUtils.js";

// Hooks for the Ultra HD renderer (src/hdRenderer.js)
export default definePatch(({ replaceCode, insertCode }) => {

    // In-game map: terrain + territory are drawn by the HD renderer when it is available
    replaceCode(`function drawGame() {
        a.b();
        ctx.setTransform(zoom, 0, 0, zoom, 0, 0);
        ctx.imageSmoothingEnabled = zoom < 3;
        ctx.drawImage(world.terrain, camera.offsetX(), camera.offsetY());
        ships.layer.draw();
        ctx.drawImage(territory, camera.offsetX(), camera.offsetY());
        /*...*/
    }`, `function drawGame() {
        a.b();
        ctx.setTransform(zoom, 0, 0, zoom, 0, 0);
        ctx.imageSmoothingEnabled = zoom < 3;
        if (__fx.hd.drawMap(ctx, zoom, zoom * camera.offsetX(), zoom * camera.offsetY(), world.terrain, territory)) ships.layer.draw();
        else {
            ctx.drawImage(world.terrain, camera.offsetX(), camera.offsetY());
            ships.layer.draw();
            ctx.drawImage(territory, camera.offsetX(), camera.offsetY());
        }
        /*...*/
    }`);

    // Main menu background map
    replaceCode(`this.drawBackground = function() {
        if (!world.ready) { ctx.fillStyle = colors.background; ctx.fillRect(0, 0, screen.w, screen.h); return; }
        var sx = screen.w / world.mapW;
        var sy = screen.h / world.mapH;
        var scale = sx > sy ? sx : sy;
        ctx.setTransform(scale, 0, 0, scale, Math.floor((screen.w - scale * world.mapW) / 2), Math.floor((screen.h - scale * world.mapH) / 2));
        ctx.drawImage(world.terrain, 0, 0);
        /*...*/
    }`, `this.drawBackground = function() {
        if (!world.ready) { ctx.fillStyle = colors.background; ctx.fillRect(0, 0, screen.w, screen.h); return; }
        var sx = screen.w / world.mapW;
        var sy = screen.h / world.mapH;
        var scale = sx > sy ? sx : sy;
        ctx.setTransform(scale, 0, 0, scale, Math.floor((screen.w - scale * world.mapW) / 2), Math.floor((screen.h - scale * world.mapH) / 2));
        __fx.hd.drawMap(ctx, scale, Math.floor((screen.w - scale * world.mapW) / 2), Math.floor((screen.h - scale * world.mapH) / 2), world.terrain, null) || ctx.drawImage(world.terrain, 0, 0);
        /*...*/
    }`);

    // Give the renderer direct access to the territory pixel data (exact bytes, including the
    // player ids that the game packs into the low bits of every pixel)
    insertCode(`function initTerritory() {
        if (territory === undefined) { territory = document.createElement("canvas"); }
        territory.width = world.mapW; territory.height = world.mapH;
        tctx = territory.getContext("2d", { alpha: true });
        tdata = tarr = null;
        tdata = tctx.getImageData(0, 0, world.mapW, world.mapH);
        tarr = tdata.data;
        util.obj.fn(tarr);
        /* here */
    }`, `__fx.hd.setTerritory(territory, tdata);`);

    // Canvas resolution: allow native and supersampled 4K rendering instead of the game's
    // default, which renders below native resolution on large screens
    insertCode(`function computeScale(w, h) {
        var avg = (w + h) / 2;
        var base = (env.id !== 0 || w < h) ? 700 : 1200;
        var s = Math.min(base / avg, 1);
        if (store.buffer.data[1].value === 0) { s = 2 * s / 3; }
        else { s = Math.min(s + (store.buffer.data[1].value - 1) * (1 - s) / 2, 1); }
        /* here */
        screen.pixelScale = (window.devicePixelRatio || 1) * s;
    }`, `s = __fx.hd.adjustResolution(s, w, h);`);
});
