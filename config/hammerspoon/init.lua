
-- voice-router hotkey — press Right Option to record, release to stop
-- Uses warm listen daemon for instant mic activation
local voiceRouter = {}
voiceRouter.active = false
voiceRouter.optDown = false
voiceRouter.stateFile = os.getenv("HOME") .. "/.local/share/voice-router/daemon-state.json"
voiceRouter.pidFile = os.getenv("HOME") .. "/.local/share/voice-router/listen-daemon.pid"
voiceRouter.pollTimer = nil
voiceRouter.alertId = nil

local function getDaemonPid()
    local f = io.open(voiceRouter.pidFile, "r")
    if not f then return nil end
    local pid = f:read("*a"):match("^%s*(.-)%s*$")
    f:close()
    if pid and #pid > 0 then return pid end
    return nil
end

local function readState()
    local f = io.open(voiceRouter.stateFile, "r")
    if not f then return nil end
    local raw = f:read("*a")
    f:close()
    local ok, data = pcall(hs.json.decode, raw)
    if ok then return data end
    return nil
end

local function signalDaemon(sig)
    local pid = getDaemonPid()
    if pid then
        hs.execute("kill -" .. sig .. " " .. pid)
        return true
    else
        hs.alert.show("✗ Listen daemon not running\nRun: voice-listen-daemon", nil, nil, 3)
        return false
    end
end

local function routeText(text)
    if not text or #text == 0 then
        hs.alert.show("(no speech detected)", nil, nil, 1.5)
        return
    end

    local env = {
        PATH = os.getenv("HOME") .. "/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        HOME = os.getenv("HOME"),
    }

    local task = hs.task.new(
        os.getenv("HOME") .. "/.local/bin/voice-route",
        function(exitCode, stdout, stderr)
            if exitCode == 0 and stdout and #stdout > 0 then
                hs.alert.show("✓ " .. stdout:sub(1, 80), nil, nil, 2)
            else
                hs.alert.show("✗ Route failed", nil, nil, 2)
            end
        end,
        {"--text", text}
    )
    task:setEnvironment(env)
    task:start()
end

local function pollForResult()
    -- Poll daemon state until transcription is done
    if voiceRouter.pollTimer then voiceRouter.pollTimer:stop() end
    voiceRouter.pollTimer = hs.timer.doEvery(0.1, function()
        local state = readState()
        if not state then return end

        if state.state == "done" then
            voiceRouter.pollTimer:stop()
            voiceRouter.pollTimer = nil
            voiceRouter.active = false
            routeText(state.text)
        elseif state.state == "error" then
            voiceRouter.pollTimer:stop()
            voiceRouter.pollTimer = nil
            voiceRouter.active = false
            hs.alert.show("✗ " .. (state.error or "unknown error"), nil, nil, 2)
        end
    end)
end

voiceRouter.tap = hs.eventtap.new({hs.eventtap.event.types.flagsChanged}, function(event)
    local flags = event:getRawEventData().CGEventData.flags
    local optPressed = (flags & 0x80000) ~= 0
    local ctrlPressed = (flags & 0x40000) ~= 0
    local cmdPressed = (flags & 0x100000) ~= 0
    local shiftPressed = (flags & 0x20000) ~= 0

    local optOnly = optPressed and not ctrlPressed and not cmdPressed and not shiftPressed

    if optOnly and not voiceRouter.optDown then
        -- Option pressed — start recording immediately
        voiceRouter.optDown = true

        if not voiceRouter.active then
            voiceRouter.active = true
            if signalDaemon("USR1") then
                -- Poll until state becomes "recording" to show alert
                local checkTimer
                checkTimer = hs.timer.doEvery(0.02, function()
                    local state = readState()
                    if state and state.state == "recording" then
                        checkTimer:stop()
                        hs.alert.show("🎤 Listening...", nil, nil, 10)
                    end
                end)
                -- Safety: stop polling after 2s
                hs.timer.doAfter(2, function() checkTimer:stop() end)
            end
        end

    elseif not optPressed and voiceRouter.optDown then
        -- Option released — stop recording, begin transcription
        voiceRouter.optDown = false
        hs.alert.closeAll()

        if voiceRouter.active then
            signalDaemon("USR2")
            hs.alert.show("⏳ Transcribing...", nil, nil, 10)
            pollForResult()
        end
    end

    return false
end)

voiceRouter.tap:start()
print("[voice-router] loaded — press Option to record")
